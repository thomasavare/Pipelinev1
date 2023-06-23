#!/usr/bin/env python

import warnings
import argparse
from time import sleep

from tensorflow import get_logger
from asr_whisper import load_whisper, asr_recording
from distilbert_classifcation import load_bert, classify
from autorecord import recording, recording_save


def language(lg):
    if lg.lower() in "french":
        return "french"
    if lg.lower() in "italian":
        return "italian"
    else:
        raise ValueError("language must be 'french'/'fr' or 'italian'/'it'")


if __name__ == "__main__":
    # parsing language if needed
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--language', default='french', help="select language between 'fr'/'french' and "
                                                                  "'it'/'italian")
    parser.add_argument('-s', '--size', default='base', help="whisper size, base or small recommended")
    parser.add_argument('-f', '--filename', default=True, help="output audiofile name")
    parser.add_argument('-s', '--seconds', default=5, help="duration of audio recording", type=int)
    parser.add_argument('-d', '--device', default=0, help="device to record on, type 'python -m sounddevice' to have"
                                                          "the list of available devices", type=int)

    args = parser.parse_args()
    language = language(args.language)

    # Many warings are undersired
    warnings.filterwarnings("ignore")
    get_logger().setLevel('INFO')

    # Loading whisper
    processor, asr_model, forced_decoder_ids = load_whisper(args.language, args.size)
    # tokenizer, cls_model = load_bert()

    sr = 16000

    # asr using microphone, by default 5 seconds
    print("recording starts in 2 seconds")
    sleep(2)
    print("start")
    if args.filename:
        myrecording = recording(sr, args.device, 16000, args.seconds)[0]
    else:
        myrecording = recording_save(sr, args.device, 16000, args.seconds, args.filename)
    print("end")

    text = asr_recording(myrecording, processor, asr_model, forced_decoder_ids)
    print(text)

    # Loading classification
    tokenizer, cls_model = load_bert()

    print(classify(text, tokenizer, cls_model, prob=True))

