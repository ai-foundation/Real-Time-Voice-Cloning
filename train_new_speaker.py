import numpy as np
from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
import librosa
import argparse
import torch
import sys
from scipy.io import wavfile
from pathlib import Path
import re
import json
import os

#####################################################################################
#  Helper methods 
#####################################################################################

#TODO: config loading functions belong in a helper file
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

# TODO: test if creating embedding even needs gpu
# check system and exit if system is lacking gpu compatability
def system_check():
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

def absoluteFilePaths(directory):
    fpaths = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            fpaths.append(os.path.abspath(os.path.join(dirpath, f)))
    print(fpaths)
    return fpaths

def confirm_parameters(speaker, audio_fpath):
    print("speaker name: ")
    print(speaker)
    print("audio clips directory: ")
    print(audio_fpath)
    print("Enter [y/n] to continue: ")
    choice = input().lower()
    if choice == "y":
        print("Continuing embedding creation...")
    else:
        print("Exiting...")
        sys.exit()

#####################################################################################
#  Train a new speaker by using average speaker embedding
#####################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path, 
                        default="encoder/saved_models/pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-f", "--audio_clips_fpath", type=Path,
                        default="",
                        help="Path to audio clip to train")
    parser.add_argument(
                    '-c', '--config_path', type=str, help='path to config file for training')
    parser.add_argument(
                    '-speaker', '--speaker_name', type=str, help='name of speaker for embedding')
    
    args = parser.parse_args()
    print_args(args, parser)
    config = load_config(args.config_path)

    embeddings_location = config.embeddings_location

    confirm_parameters(args.speaker_name, args.audio_clips_fpath)

    system_check()

    print("Preparing the encoder...")
    encoder.load_model(args.enc_model_fpath)

    audio_clip_fpaths = absoluteFilePaths(args.audio_clips_fpath)

    embed_array = None
    for audio_clip_fpath in audio_clip_fpaths:
        embed_array = None
        original_wav, sampling_rate = librosa.load(audio_clip_fpath)
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        # embed = np.expand_dims(encoder.embed_utterance(preprocessed_wav), axis=1)
        embed = encoder.embed_utterance(preprocessed_wav)
        if embed_array is None:
            embed_array = embed
        else:
            #embed_array = np.concatenate((embed_array, embed), axis = 1)
            embed_array = np.concatenate(embed_array, embed)
    embed_array

    save_embedding_to_disk(args.speaker_name, embed_array)


