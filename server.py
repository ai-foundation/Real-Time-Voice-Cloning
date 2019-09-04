from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import argparse
import torch
import sys
from flask import Flask, Response, request, render_template, send_file, url_for
from flask import jsonify
import re
import json
import base64
from scipy.io import wavfile
import io
from os import listdir
from os.path import isfile, join

#####################################################################################
#  Helper methods
#####################################################################################

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_config(config_path):
    config = AttrDict()
    with open(config_path, "r") as f:
        input_str = f.read()
    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    data = json.loads(input_str)
    config.update(data)
    return config

def save_embedding_to_disk(name, embedding):
    # TODO: move the location of the embeddings to config or as an argument
    np.save('/home/jonathan/voice-cloning-embeddings/%s.npy' % name, embedding)

def get_saved_embedding_names():
    # TODO: LOCATION OF EMBEDDINGS SHOULD COME FROM CONFIG OR ARGUMENTS
    filenames = [f for f in listdir(embeddings_location) if f.endswith('.npy')]
    return filenames

def get_speaker_embedding(speaker):
    embed = np.load(embeddings_location + '/' + speaker + '.npy')
    return embed

def trim_silence(y):
    yt, index = librosa.effects.trim(y)
    splits = []
    for start, end in librosa.effects.split(yt):
        splits.append(yt[start:end])
    yhat = np.concatenate(splits)
    return yhat

#####################################################################################
#  Initialize server and any global variables
#####################################################################################

# embeds = []
embeddings_location = '/home/jonathan/voice-cloning-embeddings'
saved_embeddings = get_saved_embedding_names()
app = Flask(__name__)

#####################################################################################
#  API routes and handlers
#####################################################################################

# TODO: Add a UI for server
@app.route('/')
def index():
    return render_template('index.html', static_url_path='/static')

@app.route('/api/speakers', methods=['GET'])
def speakers():
    speakers = get_saved_embedding_names()
    return jsonify({ "speakers": speakers })

# TODO: route for new speaker
# @app.route('/api/train', methods=['POST'])
# def train():
#     speaker = request.args.get('speaker')
#     data = request.args.get('data')
#     if (len(speaker) > 0 and data):
#         print(speaker)
#         print(data)

@app.route('/api/tts', methods=['GET'])
def tts():
    text = request.args.get('text')
    speaker = request.args.get('speaker')

    # - Directly load from the filepath:
    # TODO: should not be hardcoded
    texts = [text]
    embed = get_speaker_embedding(speaker)
    embeds = [embed]
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    # create mel spectrogram
    spec = specs[0]
    # generate waveform
    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

    out = io.BytesIO()

    wav_norm = generated_wav * (32767 / max(0.01, np.max(np.abs(generated_wav))))

    wav_norm_trimmed = trim_silence(wav_norm)

    wavfile.write(out, synthesizer.sample_rate, wav_norm_trimmed.astype(np.int16))

    data64 = base64.b64encode(out.getvalue()).decode()
    return jsonify({ "wav64": data64, "text": text })

#####################################################################################
#  Main
#####################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path, 
                        default="encoder/saved_models/pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_dir", type=Path, 
                        default="synthesizer/saved_models/logs-pretrained/",
                        help="Directory containing the synthesizer model")
    parser.add_argument("-v", "--voc_model_fpath", type=Path, 
                        default="vocoder/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--low_mem", action="store_true", help=\
        "If True, the memory used by the synthesizer will be freed after each use. Adds large "
        "overhead but allows to save some GPU memory for lower-end GPUs.")
    
    parser.add_argument(
        '-c', '--config_path', type=str, help='path to config file for training')

    args = parser.parse_args()
    print_args(args, parser)
    config = load_config(args.config_path)

    ## Print some environment information (for debugging purposes)
    print("Running a test of your configuration...\n")
    if not torch.cuda.is_available():
        print("Your PyTorch installation is not configured to use CUDA. If you have a GPU ready "
                "for deep learning, ensure that the drivers are properly installed, and that your "
                "CUDA version matches your PyTorch installation. CPU-only inference is currently "
                "not supported.", file=sys.stderr)
        quit(-1)
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" % 
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))


    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"), low_mem=args.low_mem)
    vocoder.load_model(args.voc_model_fpath)

    # TODO: THIS SHUOLD BE MOVED TO ITS OWN ROUTE
    # default_in_fpath = "/home/jonathan/venky-1.wav"
    # preprocessed_wav = encoder.preprocess_wav(default_in_fpath)
    # embed = encoder.embed_utterance(preprocessed_wav)
    # embeds = [embed]
    # save_embedding_to_disk("venky-1", embed)

    app.run(debug=True, host='0.0.0.0', port=config.port)
