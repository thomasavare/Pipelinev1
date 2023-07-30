#!/usr/bin/env python

import warnings
import argparse
from tensorflow import get_logger

from asr_whisper import load_whisper, asr, asr_recording
from distilbert_classifcation import load_bert, classify, classify_id


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

def pipeline_perf(array, **kwargs):
    text = asr_recording(array, kwargs["processor"], kwargs["asr_model"], kwargs["forced_decoder_ids"])
    return classify_id(text, kwargs["tokenizer"], kwargs["cls_model"])


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
    args = parser.parse_args()
    language = language(args.language)

    # Many warings are undersired
    warnings.filterwarnings("ignore")
    get_logger().setLevel('INFO')

    # Loading whisper and classification model
    processor, asr_model, forced_decoder_ids = load_whisper(args.language, args.size)
    tokenizer, cls_model = load_bert()
    file_name = 0

    # Processing audiofiles until we say stop
    while file_name != "exit":
        file_name = input("audiofile name: ")
        if file_name.lower() == "exit":
            continue
        res = pipeline(file_name, prob=True, processor=processor, asr_model=asr_model, forced_decoder_ids=forced_decoder_ids,
                       tokenizer=tokenizer, cls_model=cls_model)
        print(res)

