#!/usr/bin/env python

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import warnings

def load_whisper(language='french', size='base'):
    """
    load model and processor
    :param language: language for asr, italian or french to english
    :return: processor, whsiper model, forced_decoder_ids
    """
    processor = WhisperProcessor.from_pretrained("openai/whisper-"+size)
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-"+size)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="translate")
    return processor, model, forced_decoder_ids


def asr(file_name, processor, model, forced_decoder_ids):
    """
    return transcription from audio-file in French in english
    :param file_name: name of the file (any format in theory)
    :return: Audio transcription of the file.
    """
    # load mp3 file
    array, sampling_rate = librosa.load(file_name, sr=16000)
    input_features = processor(array, sampling_rate=sampling_rate, return_tensors="pt").input_features
    # generate token ids
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    # print(transcription)
    return transcription[0]


if __name__ == "__main__":
    file_name = input("file name: ")
    warnings.filterwarnings("ignore")
    processor, model, forced_decoder_ids = load_whisper()
    print("\ntranscript: ", asr(file_name, processor, model, forced_decoder_ids))
