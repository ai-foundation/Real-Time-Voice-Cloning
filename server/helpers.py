import re
import json
from os import listdir, remove
import torch
import numpy as np
import subprocess
import librosa

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

def get_saved_embedding_names(embeddings_location):
    filenames = [f for f in listdir(embeddings_location) if f.endswith('.npy')]
    return filenames

def get_speaker_embedding(speaker, embeddings_location):
    embed = np.load(embeddings_location + '/' + speaker + '.npy')
    return embed

def print_cuda_debug():
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

def create_new_speaker_embedding(speaker, filename, transcriptFileLocation, train_new_speaker_script_location, voice_clips_location):
    status = 1
    if not transcriptFileLocation:
        status = subprocess.call(["python", train_new_speaker_script_location, "-c", "conf.json", "--enc_model_fpath=/home/jonathan/rt-voice-cloning-models/encoder/saved_models/pretrained.pt",
        "--speaker_name=" + speaker, "--audio_fpath=" + voice_clips_location + "/" + filename, "--from_api=True" ])
    else:
        status = subprocess.call(["python", train_new_speaker_script_location, "-c", "conf.json", "--enc_model_fpath=/home/jonathan/rt-voice-cloning-models/encoder/saved_models/pretrained.pt",
        "--speaker_name=" + speaker, "--audio_fpath=" + voice_clips_location + "/" + filename, "--from_api=True", "--transcript_fpath=" + transcriptFileLocation])
    if (status != 0):
        abort(400, "Error creating new speaker embedding")

def denoise_output(rnnoise_script_location, tmp_dir, audio_fpath):
    raw_pcm_location = tmp_dir + "/raw.pcm"
    subprocess.call(["ffmpeg", "-i", audio_fpath, "-f", "s16le", "-acodec", "pcm_s16le", raw_pcm_location])
    rnnoise = "." + rnnoise_script_location
    denoised_pcm_location = tmp_dir + "/denoised.pcm"
    subprocess.call(["sh", rnnoise_script_location, raw_pcm_location, denoised_pcm_location])
    denoised_wav_location = tmp_dir + "/denoised.wav"
    subprocess.call(["ffmpeg", "-f", "s16le", "-ar", "16k", "-ac", "1", "-i", denoised_pcm_location, denoised_wav_location])
    return denoised_wav_location

def trim_silence(y):
    yt, index = librosa.effects.trim(y)
    splits = []
    for start, end in librosa.effects.split(yt):
        splits.append(yt[start:end])
    yhat = np.concatenate(splits)
    return yhat