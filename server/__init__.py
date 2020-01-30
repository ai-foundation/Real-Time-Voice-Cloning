from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import argparse
import torch
from flask import Flask, Response, request, render_template, send_file, url_for
from flask import jsonify, abort
from flask_cors import CORS
from server.handler import ServerHandler
from server.helpers import load_config, get_saved_embedding_names, print_cuda_debug

def initialize():
    print("Initializing fastsynth server...")
    # parser = argparse.ArgumentParser(
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter
    # )
    # parser.add_argument("-e", "--enc_model_fpath", type=Path, 
    #                     default="encoder/saved_models/pretrained.pt",
    #                     help="Path to a saved encoder")
    # parser.add_argument("-s", "--syn_model_dir", type=Path, 
    #                     default="synthesizer/saved_models/logs-pretrained/",
    #                     help="Directory containing the synthesizer model")
    # parser.add_argument("-v", "--voc_model_fpath", type=Path, 
    #                     default="vocoder/saved_models/pretrained/pretrained.pt",
    #                     help="Path to a saved vocoder")
    # parser.add_argument("--low_mem", action="store_true", help=\
    #     "If True, the memory used by the synthesizer will be freed after each use. Adds large "
    #     "overhead but allows to save some GPU memory for lower-end GPUs.")
    
    # parser.add_argument(
    #     '-c', '--config_path', type=str, help='path to config file for training')

    # args = parser.parse_args()
    # print_args(args, parser)
    config = load_config('conf.json')

    print_cuda_debug()

    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    encoder.load_model(Path(config.enc_model_fpath))
    synthesizer = Synthesizer(Path(config.syn_model_fpath).joinpath("taco_pretrained"), low_mem=False)
    vocoder.load_model(Path(config.voc_model_fpath))

    server_handler = ServerHandler(config, encoder, synthesizer, vocoder)

    app = Flask(__name__)
    app = CORS(app)
    app.add_url_rule('/', view_func=server_handler.index)
    app.add_url_rule('/api/speakers', view_func=server_handler.speakers, methods=['GET'])
    app.add_url_rule('/api/train', view_func=server_handler.train, methods=['POST'])
    app.add_url_rule('/api/tts', view_func=server_handler.tts, methods=['GET'])

    return app
