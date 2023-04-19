#%%
import pandas as pd
import numpy as np
from transformers import Wav2Vec2Tokenizer

#%%
import torchaudio
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
# %%
# Load audio waveform
waveform, sample_rate = torchaudio.load("ravdess/Actor_01/03-01-01-01-01-01-01.wav")
# %%
waveform
# %%
sample_rate
# %%
# Resample audio to 16kHz
resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)

# Convert waveform to tensor
input_values = tokenizer(waveform.numpy(), return_tensors="pt").input_values