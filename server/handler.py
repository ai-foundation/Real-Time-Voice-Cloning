# from encoder.params_model import model_embedding_size as speaker_embedding_size
# from utils.argutils import print_args
# from synthesizer.inference import Synthesizer
# from encoder import inference as encoder
# from vocoder import inference as vocoder
# from pathlib import Path
import numpy as np
import librosa
# import argparse
# import torch
# import sys
from flask import Flask, Response, request, render_template, send_file, url_for
from flask import jsonify, abort
# import re
# import json
import base64
from scipy.io import wavfile
import io
from os import listdir, remove
# from os.path import isfile, join
import time
import subprocess
from server.helpers import get_saved_embedding_names, create_new_speaker_embedding, get_speaker_embedding, denoise_output, trim_silence

class ServerHandler:
    def __init__(self, config, encoder, synthesizer, vocoder):
        self.embeddings_location = ''
        self.encoder = encoder
        self.synthesizer = synthesizer
        self.vocoder = vocoder
        self.embeddings_location = config.embeddings_location
        self.tmp_location = config.tmp_location
        self.rnnoise_script_location = config.rnnoise_script_location
        self.voice_clips_location = config.voice_clips_location
        self.train_new_speaker_script_location = config.train_new_speaker_script_location

    def index(self):
        return render_template('index.html', static_url_path='/static')

    def speakers(self):
        speakers = sorted(get_saved_embedding_names(self.embeddings_location))
        return jsonify({ "speakers": speakers })

    def train(self):
        speaker = request.args.get('speaker')
        print(speaker)
        filename = request.args.get('filename')
        print(filename)
        useTranscript = request.args.get('transcript')

        transcriptFileLocation = ''
        if useTranscript:
            transcriptFile = request.files['transcript']
            if transcriptFile:
                transcriptFileName = speaker + '-' + filename + '-transcript.txt'
                transcriptFileLocation = self.voice_clips_location + '/' + transcriptFileName
                transcriptFile.save(transcriptFileLocation)
        
        audioFile = request.files['audioFile']
        # # TODO: ALLOWED FILES METHOD
        if audioFile:
            audioFile.save(self.voice_clips_location + '/' + filename)
            create_new_speaker_embedding(speaker, filename, transcriptFileLocation, self.train_new_speaker_script_location, self.voice_clips_location)

            if transcriptFileLocation:
                remove(transcriptFileLocation)

            return jsonify({"status": "ok"})

        abort(400, "Invalid audio file")

    def tts(self):
        text = request.args.get('text')
        speaker = request.args.get('speaker')
        denoise = request.args.get('denoise')

        # - Load the speaker embedding:
        texts = [text]
        embed = get_speaker_embedding(speaker, self.embeddings_location)
        embeds = [embed]
        specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
        # create mel spectrogram
        spec = specs[0]
        # generate waveform
        generated_wav = self.vocoder.infer_waveform(spec)
        generated_wav = np.pad(generated_wav, (0, self.synthesizer.sample_rate), mode="constant")

        if (denoise == 'True'):
            timestamp = str(time.time()).replace('.', '-')
            tmp_dir = self.tmp_location + '/' + speaker + '-' + timestamp
            subprocess.call(["mkdir", tmp_dir])
            tmp_fpath = tmp_dir + '/' + speaker + '-' + timestamp + '.wav'
            librosa.output.write_wav(tmp_fpath, generated_wav.astype(np.float32), self.synthesizer.sample_rate)
            
            denoised_output_fpath = denoise_output(self.rnnoise_script_location, tmp_dir, tmp_fpath)
            y, sr = librosa.load(denoised_output_fpath, sr=16000)
            generated_wav = y

        wav_norm = generated_wav * (32767 / max(0.01, np.max(np.abs(generated_wav))))
        wav_norm_trimmed = trim_silence(wav_norm)
        out = io.BytesIO()
        wavfile.write(out, self.synthesizer.sample_rate, wav_norm_trimmed.astype(np.int16))

        data64 = base64.b64encode(out.getvalue()).decode()
        return jsonify({ "wav64": data64, "text": text })
