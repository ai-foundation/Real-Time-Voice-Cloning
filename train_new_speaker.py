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
from aeneas.exacttiming import TimeValue
from aeneas.executetask import ExecuteTask
from aeneas.language import Language
from aeneas.syncmap import SyncMapFormat
from aeneas.task import Task
from aeneas.task import TaskConfiguration
from aeneas.textfile import TextFileFormat
import aeneas.globalconstants as gc
from pydub import AudioSegment

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

def save_embedding_to_disk(name, embedding, embeddings_location):
    np.save(embeddings_location + '/%s.npy' % name, embedding)

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

def load_audio_file(audio_fpath):
    # split = audio_fpath.suffix
    # ftype = split[len(split) - 1]
    ftype = audio_fpath.suffix
    audio = None
    if (ftype == '.wav'):
        audio = AudioSegment.from_wav(audio_fpath)
    elif (ftype == '.mp3'):
        audio = AudioSegment.from_mp3(audio_fpath)
    elif (ftype == '.ogg'):
        audio = AudioSegment.from_ogg(audio_fpath)
    elif (ftype == '.aac'):
        audio = AudioSegment.from_aac(audio_fpath)
    return audio

def split_audio_into_clips(audio_fpath, speaker_name, transcript_fpath, voice_clips_location):
    # create Task object
    config = TaskConfiguration()
    config[gc.PPN_TASK_LANGUAGE] = Language.ENG
    config[gc.PPN_TASK_IS_TEXT_FILE_FORMAT] = TextFileFormat.PLAIN
    config[gc.PPN_TASK_OS_FILE_FORMAT] = SyncMapFormat.JSON
    task = Task()
    task.configuration = config
    task.audio_file_path_absolute = audio_fpath
    task.text_file_path_absolute = transcript_fpath

    print("===> Calculating force alignment ...")
    ExecuteTask(task).execute()
    full_voice_clip = load_audio_file(audio_fpath)
    if (full_voice_clip == None):
        print('Unknown audio file extension')
        sys.exit('Unknown audio file extension')

    os.mkdir(voice_clips_location + "/" + speaker_name, mode = 0o777)

    print("===> Splitting clips using force alignment boundaries ...")
    index = 0
    for frag in task.sync_map.fragments:
        if (index != 0 and index != 6):
            begin = frag.begin * 1000
            end = frag.end * 1000
            clip = full_voice_clip[float(begin): float(end)]
            clip_name = speaker_name + '-' + str(index)
            clip.export(voice_clips_location + "/" + speaker_name + "/" + clip_name + ".wav", format="wav")
        index = index + 1

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
    parser.add_argument("-f", "--audio_fpath", type=Path,
                        default="",
                        help="Path to audio clip to train")
    parser.add_argument(
                    '-c', '--config_path', type=str, help='path to config file for training')
    parser.add_argument(
                    '-speaker', '--speaker_name', type=str, help='name of speaker for embedding')
    parser.add_argument(
                    '-transcript', '--transcript_fpath', type=str, help='path to audio clip transcript')
    parser.add_argument(
                    '-api', '--from_api', type=str, help='boolean whether script was called from api')


    args = parser.parse_args()
    print_args(args, parser)
    config = load_config(args.config_path)

    embeddings_location = config.embeddings_location
    voice_clips_location = config.voice_clips_location

    if (args.from_api is None):
        confirm_parameters(args.speaker_name, args.audio_fpath)
        system_check()

    print("Preparing the encoder...")
    encoder.load_model(args.enc_model_fpath)

    embed_array = None
    if args.transcript_fpath:
        # takes the audio clip of the speaker and splits it up into multiple shorter audio clips
        split_audio_into_clips(args.audio_fpath, args.speaker_name, args.transcript_fpath, voice_clips_location)

        audio_clip_fpaths = absoluteFilePaths(voice_clips_location + "/" + args.speaker_name)

        print("===> Processing clips and saving speaker embedding ...")
        for audio_clip_fpath in audio_clip_fpaths:
            embed_array = None
            original_wav, sampling_rate = librosa.load(audio_clip_fpath)
            preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
            embed = encoder.embed_utterance(preprocessed_wav)
            if embed_array is None:
                embed_array = embed
            else:
                embed_array = np.concatenate(embed_array, embed)
    else:
        original_wav, sampling_rate = librosa.load(args.audio_fpath)
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        embed = encoder.embed_utterance(preprocessed_wav)
        embed_array = embed

    save_embedding_to_disk(args.speaker_name, embed_array, embeddings_location)
    print("DONE! Speaker embedding for - " + args.speaker_name + " saved to " + embeddings_location)
