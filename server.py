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
import time
import subprocess

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

def get_saved_embedding_names():
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

def denoise_output(rnnoise_script_location, tmp_dir, audio_fpath):
    print("========> Beginning denoise process...")
    raw_pcm_location = tmp_dir + "/raw.pcm"
    subprocess.call(["ffmpeg", "-i", audio_fpath, "-f", "s16le", "-acodec", "pcm_s16le", raw_pcm_location])
    rnnoise = "." + rnnoise_script_location
    denoised_pcm_location = tmp_dir + "/denoised.pcm"
    subprocess.call(["sh", rnnoise_script_location, raw_pcm_location, denoised_pcm_location])
    denoised_wav_location = tmp_dir + "/denoised.wav"
    subprocess.call(["ffmpeg", "-f", "s16le", "-ar", "16k", "-ac", "1", "-i", denoised_pcm_location, denoised_wav_location])
    return denoised_wav_location

#####################################################################################
#  Define server and other global variables
#####################################################################################

embeddings_location = ''
saved_embeddings = ''
tmp_location = ''
rnnoise_script_location = ''
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
    speakers = sorted(get_saved_embedding_names())
    return jsonify({ "speakers": speakers })

# TODO: route for new speaker
@app.route('/api/train', methods=['POST'])
def train():
    speaker = request.args.get('speaker')
    print(speaker)

    # payload = request.data
    # print(payload)

    # f = open('/home/jonathan/voice-clips/upload-test.wav', 'wb')
    # f.write(payload)
    # f.close()

    # file = request.files['file']
    print(request.files)
    file = request.files['file']
    # # TODO: ALLOWED FILES METHOD
    if file:
        print(file)
        file.save('/home/jonathan/voice-clips/upload-test.wav')

    return jsonify({"status": "ok"})

@app.route('/api/tts', methods=['GET'])
def tts():
    text = request.args.get('text')
    speaker = request.args.get('speaker')
    denoise = request.args.get('denoise')

    # - Load the speaker embedding:
    texts = [text]
    embed = get_speaker_embedding(speaker)
    embeds = [embed]
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    # create mel spectrogram
    spec = specs[0]
    # generate waveform
    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

    if (denoise == 'True'):
        timestamp = str(time.time()).replace('.', '-')
        tmp_dir = tmp_location + '/' + speaker + '-' + timestamp
        subprocess.call(["mkdir", tmp_dir])
        tmp_fpath = tmp_dir + '/' + speaker + '-' + timestamp + '.wav'
        librosa.output.write_wav(tmp_fpath, generated_wav.astype(np.float32), synthesizer.sample_rate)
        
        # WIP
        denoised_output_fpath = denoise_output(rnnoise_script_location, tmp_dir, tmp_fpath)
        y, sr = librosa.load(denoised_output_fpath, sr=16000)
        generated_wav = y

    wav_norm = generated_wav * (32767 / max(0.01, np.max(np.abs(generated_wav))))
    wav_norm_trimmed = trim_silence(wav_norm)
    out = io.BytesIO()
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

    embeddings_location = config.embeddings_location
    tmp_location = config.tmp_location
    rnnoise_script_location = config.rnnoise_script_location
    saved_embeddings = get_saved_embedding_names()

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

    print("Starting fastsynth server...")
    app.run(debug=True, host='0.0.0.0', port=config.port)
