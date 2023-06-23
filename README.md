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

The whisper model we're using is "whisper-small" with 244 millions parameters, but we could use "whisper-base" which has 
74 millions parameters but is not as effective as whispe-small.

To use whisper, we have to load:

- The processor
- The model
- The forced encoder ids

(Will have to so more research on the purpose of these things)

From now, we have to do the asr task with audio files because I couldn't figure out how to use the microphone (may not 
work on my computer). ```python-sounddevice``` library seems to not work. I accidentally deleted my test file.

We don't need to have a continuous asr task which would use a queue but it could be useful if we want an automatic 
starting/ending audio recognition. I will try to implement it if i can make the microphone work.

**by default, whisper is set for french-to-english, but it can be modified (see usage)**

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
### Pipeline 1

Using conda, create the environment with env.yml file

```conda create -n Pipelinev1 --file environment.yml```

be careful to use ```environment.yml``` and not ```env.ymel```, I forgot to delete it.

If you have issues with python-sounddevice, delete it from the virtual environment and install it with pip.

To then use the pipeline simply type:

```./pipeline.py --language [fr/french/it/italian | def french] --size [whsiper size | def base]```

I deliberately chose to only use french or italian.

Then the input named ```audio file name: ``` will appear and simply enter the audiofile path/name.

To exit the loop, simply type ```exit``` instead of an audio file.

Pipeline 1 is going to be used to evaluate the performances of the pipeline.


### Pipeline 2 & 3

Pipeline 2 and 3 are similar, the only difference is that pipeline 3 implements an autostop for the recording and 
pipeline 2 is stopped by using ```ctrl+c```

### Example for pipeline 1

Let's use one example. *coca-cola.m4a* is an audio file for the sentence "J'ai besoin de jeter une canette de coca"
which translates to "I need to throw a can of coca".

First let's start the pipeline.

```./pipeline.py -language french -size small```

After everything loaded, let's select the right audio file:

```audiofile name: coca-cola.m4a```

We have this result:

``` 
I need to throw a can of coca.
1/1 [==============================] - 1s 812ms/step
('ALUMINIUM CAN', 0.99602026)
```

Voilà !
