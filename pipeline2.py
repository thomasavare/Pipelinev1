#!/usr/bin/env python

import warnings
import argparse
from time import sleep

from tensorflow import get_logger
from asr_whisper import load_whisper, asr_recording
from distilbert_classifcation import load_bert, classify
from micro import recording


def pipeline(file_name, prob=False, **kwargs):
    """
    Full pipeline for waste classification using whsiper and distilbert (finetuned)
    :param file_name: audio file to process through pipeline
    :param asr_args: [processor
    :param classification_args:
    :return:
    """
    text = asr(file_name, kwargs["processor"], kwargs["asr_model"], kwargs["forced_decoder_ids"])
    print(text)
    return classify(text, kwargs["tokenizer"], kwargs["cls_model"], prob=prob)


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
    parser.add_argument('-language', default='french', help="select language between 'fr'/'french' and 'it'/'italian")
    parser.add_argument('-size', default='base', help="whisper size, base or small recommended")
    parser.add_argument('-filename', default='output.wav', help="output audiofile name")
    parser.add_argument('-seconds', default=5, help="duration of audio recording", type=int)
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
    myrecording = recording(args.seconds, sr, args.filename)[0]
    print("end")

    text = asr_recording(myrecording, processor, asr_model, forced_decoder_ids, sr)
    print(text)

    # Loading classification
    tokenizer, cls_model = load_bert()

    print(classify(text, tokenizer, cls_model, prob=True))

