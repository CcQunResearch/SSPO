import edge_tts
import requests
import json
import librosa
import tempfile
from pydub import AudioSegment
import uuid
import os
import soundfile as sf
import time
from concurrent.futures import ThreadPoolExecutor


def contains_chinese(text):
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


def contains_english(text):
    for char in text:
        if ('a' <= char <= 'z') or ('A' <= char <= 'Z'):
            return True
    return False


def contains_thai(text):
    for char in text:
        if '\u0e00' <= char <= '\u0e7f':
            return True
    return False


def contains_spanish(text):
    spanish_chars = set('áéíóúüñ¿¡ÁÉÍÓÚÜÑ')
    for char in text:
        if char.lower() in spanish_chars:
            return True
    return False


def trim_silence(audio_path, output_path, threshold=0.01, min_silence_duration=0.5):
    y, sr = librosa.load(audio_path, sr=None)
    frame_length = int(sr * 0.025)
    hop_length = int(sr * 0.010)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    start_frame = 0
    while start_frame < len(rms) and rms[start_frame] < threshold:
        start_frame += 1
    end_frame = len(rms) - 1
    while end_frame >= 0 and rms[end_frame] < threshold:
        end_frame -= 1
    start_sample = start_frame * hop_length
    end_sample = (end_frame + 1) * hop_length
    y_trimmed = y[start_sample:end_sample]
    sf.write(output_path, y_trimmed, sr)


def tts(TEXT):
    if contains_thai(TEXT):
        VOICE = "th-TH-NiwatNeural"
    elif contains_spanish(TEXT):
        VOICE = "es-ES-AlvaroNeural"
    elif contains_chinese(TEXT):
        VOICE = "zh-CN-XiaoyiNeural"
    else:
        punc_list = ['.', '!', '?']
        for punc in punc_list:
            TEXT = TEXT.replace(punc, ' ').replace('  ', ' ')
        VOICE = "en-US-AriaNeural"
    temp_audio_file = f"info/{uuid.uuid4()}.wav"
    duration = -1
    try:
        communicate = edge_tts.Communicate(TEXT, VOICE)
        communicate.save_sync(temp_audio_file)
        trim_silence(temp_audio_file, temp_audio_file, threshold=0.01, min_silence_duration=0.5)
        accelerated_audio = AudioSegment.from_file(temp_audio_file)
        duration = round(len(accelerated_audio) * 0.001, 3)
    finally:
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
    return duration
