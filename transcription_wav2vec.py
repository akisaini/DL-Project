#%%
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
#%%
import torchaudio
# %%
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
# %%
# Load audio waveform
waveform, sample_rate = torchaudio.load("ravdess/Actor_01/03-01-01-01-01-01-01.wav")

#%%
# Resample audio to 16kHz
resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)
#%%
# Convert waveform to tensor
input_values = tokenizer(waveform.numpy(), return_tensors="pt").input_values
#%%
input_values = input_values.reshape(1, -1)
#%%
input_values = torch.tensor(input_values)
#%%

#%%
# Run inference
with torch.no_grad():
    logits = model(input_values).logits

# Decode predicted transcription
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.decode(predicted_ids[0])
print(transcription)

#%%
