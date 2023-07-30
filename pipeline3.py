#!/usr/bin/env python

import warnings
import argparse
from time import sleep

from tensorflow import get_logger
from asr_whisper import load_whisper, asr_recording
from autorecord import autorecord, autorecord_save
from distilbert_classifcation import load_bert, classify


def language(lg):
    if lg.lower() in "french":
        return "french"
    if lg.lower() in "italian":
        return "italian"
    else:
        raise ValueError("language must be 'french'/'fr' or 'italian'/'it'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', default='french', help="select language, default='french'")
    parser.add_argument('-s', '--size', default='base', help="whisper size, base or small recommended")
    parser.add_argument('-d', '--device', default=0, help="device to use for microphone, list of devices type 'python -m sounddevice'", type=int)
    parser.add_argument('-b', '--blocksize', default=16000, help="size of recording block, by default 16000, so 1 block is one second", type=int)
    parser.add_argument('-f', '--filename', default=True, help="If wishing to save audio in a file, ex 'test.wav'")
    args = parser.parse_args()
    language = language(args.language)

    # Many warings are undersired
    warnings.filterwarnings("ignore")
    get_logger().setLevel('INFO')

    sr = 16000
    blocksize = 16000 # 1 block = 1

    # asr using microphone, max 10 seconds
    if args.filename:
        record = autorecord(sr, args.device, args.blocksize)
    else:
        record = autorecord_save(sr, args.device, args.blocksize, args.filename)

    print(record)

    # Loading whisper
    processor, asr_model, forced_decoder_ids = load_whisper(args.language, args.size)

    text = asr_recording(record, processor, asr_model, forced_decoder_ids)
    print(text, flush=True)

    # Loading classification
    tokenizer, cls_model = load_bert()

    print(classify(text, tokenizer, cls_model, prob=True))

