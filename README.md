# Pipeline, first version

Pipeline for audiofile to text classification.

## Pipeline overview

1. Loading whisper
2. Loading finetuned distilBERT
3. Entering classification loop:
   1. Load audiofile
   2. asr with Whisper
   3. classification

### Loading Whisper

The whisper model we're using is "whisper-base" with 74 millions parameters, but we could use "whisper-small" which has 
244 millions parameters and has a fairly short inference.

To use whisper, we have to load:

- The processor
- The model
- The forced encoder ids

(Will have to so more research on the purpose of these things)

From now, we have to do the asr task with audio files because I couldn't figure out how to use the microphone (may not 
work on my computer). ```python-sounddevice``` library seems to not work. I accidentally deleted my test file.

We don't need to have a continuous asr task which would use a queue but it could be useful if we want an automatic 
starting/ending audio recognition. I will try to implement it if i can make the microphone work.

**For now, whisper is set for french-to-english**

### Loading DistilBERT

Loading custom fine-tuned model through huggingface transformers library. The model is named
[```thomasavare/distilbert-ft-test3```](https://huggingface.co/thomasavare/distilbert-ft-test3). The model card is not 
updated yet.

We're loading:

- fine-tuned model
- DistilBERT tokenizer

The classification task goes as followed:

1. The text is tokenized
2. Classification
3. Extracting the results for each class from the logits layer
4. applying softmax function to convert from logits to "probabilities"
5. extracting the index with highest probability and with a dictionary extracting the corresponding class.

## Using the pipeline

Using conda, create the environment with env.yml file

```conda create -n Pipelinev1 --file env.yml```

To then use the pipeline simply type:

```./pipeline.py```

Then the input named ```audio file name: ``` will appear and simply enter the audiofile path/name.

In the next days, I will introduce an argument to switch Whisper to italian-to-english and french-to-english more
more easily than changing the code.
