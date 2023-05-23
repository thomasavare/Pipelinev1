#!/usr/bin/env python

import warnings
from tensorflow import get_logger
from asr_whisper import load_whisper, asr
from distilbert_classifcation import load_bert, classify


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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    get_logger().setLevel('INFO')
    processor, asr_model, forced_decoder_ids = load_whisper()
    tokenizer, cls_model = load_bert()
    file_name = 0
    while file_name != "exit":
        file_name = input("audiofile name: ")
        if file_name.lower() == "exit":
            continue
        res = pipeline(file_name, prob=True, processor=processor, asr_model=asr_model, forced_decoder_ids=forced_decoder_ids,
                       tokenizer=tokenizer, cls_model=cls_model)
        print(res)

