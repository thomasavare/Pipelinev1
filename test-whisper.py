from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# load model and processor

processor = WhisperProcessor.from_pretrained("openai/whisper-base")

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="translate")

# load mp3 file

array, sampling_rate = librosa.load('test-fr.wav', sr=16000)

input_features = processor(array, sampling_rate=sampling_rate, return_tensors="pt").input_features

# generate token ids

predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

# decode token ids to text

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription)
