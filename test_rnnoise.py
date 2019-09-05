import rnnoise
import os, sys
import wave
import ctypes
import contextlib
import numpy as np
from ctypes import util
from scipy.io import wavfile
from pydub import AudioSegment


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate        
        
      
def frame_generator(frame_duration_ms,
                    audio,
                    sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield audio[offset:offset + n]
        offset += n        
        
denoiser = rnnoise.RNNoise()

##########################################################################
#     Script for testing rnnoise on wav
##########################################################################

wav_path = 'oh_sorry_did_you_expect_me_to_have_the_answer_to_that_500.wav'


TARGET_SR = 16000
TEMP_FILE = 'test.wav'

sound = AudioSegment.from_wav(wav_path)
sound = sound.set_frame_rate(TARGET_SR)
sound = sound.set_channels(1)

sound.export(TEMP_FILE,
             format="wav")

audio, sample_rate = read_wave(TEMP_FILE)
assert sample_rate == TARGET_SR

frames = frame_generator(10, audio, TARGET_SR)
frames = list(frames)
tups = [denoiser.process_frame(frame) for frame in frames]
denoised_frames = [tup[1] for tup in tups]

denoised_wav = np.concatenate([np.frombuffer(frame,
                                             dtype=np.int16)
                               for frame in denoised_frames])

wavfile.write('oh_sorry_denoised.wav',
              TARGET_SR,
              denoised_wav)
