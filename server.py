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

embeds = []

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

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c', '--config_path', type=str, help='path to config file for training')
args = parser.parse_args()

config = load_config(args.config_path)
app = Flask(__name__)
enc_model_fpath = config.encoder_model_fpath
voc_model_fpath = config.vocoder_model_fpath
syn_model_fpath = config.synthesizer_model_fpath

# TODO: Add a UI for server
# @app.route('/')
# def index():
#     return render_template('index.html', static_url_path='/static')

# TODO: route for new speaker
# @app.route('/api/train', methods=['POST'])
# def train():

# TODO: route shuould take parameter for embedding
@app.route('/api/tts', methods=['GET'])
def tts():
    text = request.args.get('text')

    # - Directly load from the filepath:
    # TODO: should not be hardcoded
    texts = [text]
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    # create mel spectrogram
    spec = specs[0]
    # generate waveform
    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

    data64 = base64.b64encode(np.array2string(generated_wav))
    return jsonify({ "wav64": data64, "text": text })

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

# TODO: embedding should not be hardcoded
default_in_fpath = "/home/jonathan/jon_voice.wav"
preprocessed_wav = encoder.preprocess_wav(default_in_fpath)
embed = encoder.embed_utterance(preprocessed_wav)
embeds = [embed]

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=config.port)
