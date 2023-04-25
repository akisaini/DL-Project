# from threading import local
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import librosa
# from IPython.display import Audio #helps play sound
import os
import sys
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score
import glob

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

'''
Understanding file formatting:

Modality (01 = full-AV, 02 = video-only, 03 = audio-only). -  All files are 03.

Vocal channel (01 = speech, 02 = song). - All files are 01. 

Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).

Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.

Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").

Repetition (01 = 1st repetition, 02 = 2nd repetition).

Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
'''

'''
x, sr = librosa.load('03-01-01-01-01-01-01.wav')
sf.write('03-01-01-01-01-01-01.wav', x, sr)
#x is the audio time series
#sr is the sampling rate of x
Audio(data = x, rate = sr)
#%%
#Create waveplot of speech input
plt.figure(figsize=(8, 4))
librosa.display.waveshow(x, sr = sr)
plt.title('Waveplot - Male Neutral')
plt.show()
# %%
#Create Spectogram of speech input for further analysis
spectrogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128,fmax=8000) 
spectrogram = librosa.power_to_db(spectrogram)
librosa.display.specshow(spectrogram, y_axis='mel', fmax=8000, x_axis='time');
plt.title('Mel Spectrogram - Male Neutral')
plt.colorbar(format='%+2.0f dB');
'''


emotion = []
intensity = []
file_path = []
act_gender = []
main_path = []
origin = []

rootdir = os.getcwd() + os.path.sep + 'ravdess'
# print(rootdir)

# getting gender, emotion, actor(M/F)
for path in glob.glob(f'{rootdir}/*/**'):
    main_path.append(path)

    path_short = path.split(os.path.sep)[-1]  # getting all the audio filenames to create gender specific tabular audio data.

    file_path.append(path)

    info = path_short.split('-')
    emotion.append(int(info[2]) - 1)
    intensity.append(info[3])
    gender = int(info[-1].split('.')[0])
    act_gender.append('F' if gender % 2 == 0 else 'M')
    origin.append(True)

audio_tab = pd.DataFrame.from_dict({
    'gender': act_gender,
    'emotion': emotion,
    'intensity': intensity,
    'path': file_path,
    'origin': origin,
})

audio_tab_new = audio_tab.copy()
audio_tab_new['origin'] = False

audio_natural = audio_tab_new[(audio_tab_new['emotion'] == 0)].copy()

pd_audio = pd.concat([audio_tab, audio_natural, audio_tab_new, audio_natural, audio_tab_new, audio_natural, audio_tab_new, audio_natural])

total_num = len(pd_audio)
test_size = 0.2
test_num = int(round(total_num * test_size, 0))
train_num = total_num - test_num

print(total_num, test_num, train_num, test_size)

from sklearn.utils import shuffle
pd_audio = shuffle(pd_audio, random_state=7)

# print(list(pd_audio['emotion']))

import collections

print(collections.Counter(pd_audio['emotion']))

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.85):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def transform_audio(data, fns, sampling_rate):
    fn = random.choice(fns)
    if fn == pitch:
        fn_data = fn(data, sampling_rate)
    elif fn == "None":
        fn_data = data
    elif fn in [noise, stretch]:
        fn_data = fn(data)
    else:
        fn_data = data
    return fn_data

# get the mel-scaled spectrogram (ransform both the y-axis (frequency) to log scale,
# and the “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes.)
log_spectrogram = []
emotion_lst = []
fns = [noise, pitch, stretch, shift] # "None"

for i in range(len(pd_audio)):
    item = pd_audio.iloc[i].path
    y, sample_rate = librosa.load(item, duration=3, sr=44100)  # Load an audio file as a floating point time series.

    if pd_audio.iloc[i].origin is False:
        fn1_data = transform_audio(y, fns, sample_rate)
        y = transform_audio(fn1_data, fns, sample_rate)

    spec = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=128, fmax=8000)
    db_spec = librosa.power_to_db(spec)  # Convert a power spectrogram (amplitude squared) to decibel (dB) units
    # temporally average spectrogram
    log_spectrogram.append(np.mean(db_spec, axis=0))
    if i % 100 == 0:
        print('do preprossing', i//100)

max_row_length = max([len(row) for row in log_spectrogram])

padded_list_of_lists = []
for row in log_spectrogram:
    padding_length = max_row_length - len(row)
    if padding_length > 0:
        # avg_value = np.mean(row[4:])
        avg_value = 0
        padded_row = np.pad(row, (0, padding_length), mode='constant', constant_values=avg_value)
        padded_list_of_lists.append(padded_row[4:])
    else:
        padded_list_of_lists.append(row[4:])

# Convert the padded list of lists into a NumPy array
padded_array = np.array(padded_list_of_lists)[:, :250]


# ohe of target variable using label encoder:
lb = LabelEncoder()
# y_all = to_categorical(lb.fit_transform(np.array(emotion)))
y_all = list(pd_audio['emotion'])

# %%
# splitting into training and testing:
X_train, X_test, y_train, y_test = train_test_split(padded_array, y_all, test_size=test_size, random_state=0)


# Normalizing the data
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# print(X_train.shape)
print('X_train.shape:', X_train.shape)
# size = int(input())

# %%
# Reshape data to include 3D tensor
# Reshape and convert the input array to a PyTorch tensor
X_train = torch.tensor(X_train, dtype=torch.float).reshape(train_num, 5, 1, -1)
X_test = torch.tensor(X_test, dtype=torch.float).reshape(test_num, 5, 1, -1)

y_train = torch.tensor(y_train, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)
print('get data and ready to train:', X_train.shape, X_test.shape)

# %%
# Define the training parameters
input_size = X_train.shape[3]
hidden_size = 96
# n_categories = y_train.shape[1]
n_categories = 8

print('input_size:', input_size, '  hidden_size:', hidden_size)

class RNN(nn.Module):
    # implement RNN from scratch rather than using nn.RNN
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o1 = nn.Linear(input_size + hidden_size, 64)
        self.i2o2 = nn.Linear(64, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)

        hidden = self.i2h(combined)
        output = self.i2o1(combined)
        output = self.i2o2(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


rnn = RNN(input_size, hidden_size, n_categories)
print('the original model')

criterion = nn.NLLLoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, verbose=True)

max_epochs = 150


def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()


# %%
def predict(line_tensor, category_tensor):
    with torch.no_grad():
        hidden = rnn.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        loss = criterion(output, category_tensor)

        return output, loss.item()


train_loss_lst = []
test_loss_lst = []
acc_train_lst = []
acc_test_lst = []


# Train the RNN model
for epoch in range(1, max_epochs + 1):

    train_loss, steps_train = 0, 0
    test_loss, steps_test = 0, 0
    pred_train = []

    for i in range(X_train.size(0)):
        category_tensor = y_train[i].reshape(1).type(torch.LongTensor)
        line_tensor = X_train[i]
        output, loss = train(line_tensor, category_tensor)
        output_item = torch.argmax(output).item()
        pred_train.append(output_item)

        train_loss += loss
        steps_train += 1

    avg_train_loss = train_loss / steps_train

    pred = []
    for i in range(X_test.shape[0]):
        output, loss = predict(X_test[i], y_test[i].reshape(1).type(torch.LongTensor))
        output_item = torch.argmax(output).item()
        pred.append(output_item)

        test_loss += loss
        steps_test += 1

    avg_test_loss = test_loss / steps_test
    # scheduler.step(avg_train_loss)

    acc_train = f1_score(pred_train, y_train, average='micro')
    acc_test = f1_score(pred, y_test, average='micro')

    print('epoch', epoch, 'loss', avg_train_loss, 'f1_score_train', acc_train, 'f1_score_test', acc_test)

    train_loss_lst.append(avg_train_loss)
    test_loss_lst.append(avg_test_loss)
    acc_train_lst.append(acc_train)
    acc_test_lst.append(acc_test)

import matplotlib.pyplot as plt

epochs = [i for i in range(max_epochs)]
fig , ax = plt.subplots(1,2)

fig.set_size_inches(20,6)

ax[0].plot(epochs , train_loss_lst , label = 'Training Loss')
ax[0].plot(epochs , test_loss_lst , label = 'Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(epochs , acc_train_lst , label = 'Training Accuracy')
ax[1].plot(epochs , acc_test_lst , label = 'Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()
