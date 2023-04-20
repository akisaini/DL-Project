# %%
from threading import local
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa
from IPython.display import Audio  # helps play sound
import os
import sys
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn

# %%
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

x, sr = librosa.load('03-01-01-01-01-01-01.wav')
sf.write('03-01-01-01-01-01-01.wav', x, sr)
# x is the audio time series
# sr is the sampling rate of x
Audio(data=x, rate=sr)
# %%
# Create waveplot of speech input
plt.figure(figsize=(8, 4))
librosa.display.waveshow(x, sr=sr)
plt.title('Waveplot - Male Neutral')
plt.show()
# %%
# Create Spectogram of speech input for further analysis
spectrogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128, fmax=8000)
spectrogram = librosa.power_to_db(spectrogram)
librosa.display.specshow(spectrogram, y_axis='mel', fmax=8000, x_axis='time');
plt.title('Mel Spectrogram - Male Neutral')
plt.colorbar(format='%+2.0f dB');
# %%
emotion = []
gender = []
intensity = []
file_path = []
# %%
import glob

rootdir = 'C:\\Users\\saini\\Documents\\GWU\\6450,11-23SP-CC\\Project\\ravdess'
for main_path in glob.glob(f'{rootdir}/*/**'):
    splt = main_path.split('\\')
    print(splt[-1])  # getting all the audio filenames to create gender specific tabular audio data.
    file_path.append(splt[-1])
# %%
# getting gender, emotion, actor(M/F)
for i in range(len(file_path)):
    info = file_path[i].split('-')
    # print(info)
    emotion.append(info[2])
    intensity.append(info[3])
    gender.append(info[-1])

# %%
act_gender = []
for i in gender:
    gend = i.split('.')
    act_gender.append(gend[0])

# converting list elements from str to int
act_gender = list(map(int, act_gender))
act_gender = ['F' if i % 2 == 0 else 'M' for i in act_gender]
# %%
audio_tab = pd.DataFrame(emotion)
# %%
audio_tab.replace({1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'},
                  inplace=True)
audio_tab = pd.concat([pd.DataFrame(act_gender), audio_tab, pd.DataFrame(intensity), pd.DataFrame(file_path)], axis=1)
audio_tab.columns = ['gender', 'emotion', 'intensity', 'path']
# %%
# EXTRACTING LOG MEL SPECTROGRAM MEAN VALUES
df = pd.DataFrame(columns=['mel_spectrogram'])
counter = 0
main_path = []
for i in glob.glob(f'{rootdir}/*/**'):
    main_path.append(i)

# %%
num = []
sr_lst = []
for i in range(len(main_path)):
    X, sample_rate = librosa.load(main_path[i], duration=3, sr=44100, offset=0.5)
    num.append(X)
    sr_lst.append(sample_rate)
    # get the mel-scaled spectrogram (ransform both the y-axis (frequency) to log scale, and the “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes.)
# %%
log_spectrogram = []
for i in range(len(num)):
    spec = librosa.feature.melspectrogram(y=num[i], sr=sr_lst[i], n_mels=128, fmax=8000)
    db_spec = librosa.power_to_db(spec)
    # temporally average spectrogram
    log_spectrogram.append(np.mean(db_spec, axis=0))

df['mel_spectrogram'] = log_spectrogram

df_main = pd.concat([audio_tab, df], axis=1)

# %%
split_df = pd.DataFrame(df_main['mel_spectrogram'].tolist())

# The final tabular dataframe containing frequency in dB
df_main = pd.concat([audio_tab, split_df], axis=1)

# %%
# Imputing NaN's:
for i in range(0, 1440):
    df_main.iloc[i, 4:] = df_main.iloc[i, 4:].fillna(df_main.iloc[i, 4:].median())
# %%
# splitting into training and testing:
train_data, test_data = train_test_split(df_main, test_size=0.25, random_state=0)
# %%
X_train = train_data.iloc[:, 4:]
y_train = train_data.loc[:, 'emotion']  # train target label

X_test = test_data.iloc[:, 4:]
y_test = test_data.loc[:, 'emotion']  # test target label

# %%
# Normalizing the data
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
# %%
# converting into np array:

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
# %%
# ohe of target variable using label encoder:
lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train))
y_test = to_categorical(lb.fit_transform(y_test))
# %%
# Reshape data to include 3D tensor
X_train = X_train[:, :, np.newaxis]
X_test = X_test[:, :, np.newaxis]

# %%

# Reshape and convert the input array to a PyTorch tensor
X_train = torch.tensor(np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)), dtype=torch.float)
# %%
y_train = y_train.reshape(-1, 8)
y_train = torch.tensor(y_train, dtype=torch.float)

# %%
# Define the training parameters
input_size = X_train.shape[2]
hidden_size = 10
output_size = y_train.shape[1]
learning_rate = 0.01
max_epochs = 40
#%%
# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=4, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


# Convert the input and target arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)

# Initialize the LSTM model and the optimizer
lstm = LSTM(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# %%
# Train the LSTM model
for epoch in range(max_epochs):
    optimizer.zero_grad()
    y_pred = lstm(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, max_epochs, loss.item()))


# Epoch 30 loss: 0.355356015751954

 # %%
# Convert the validation set to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)

# Set the model to evaluation`` mode
with torch.no_grad():
    lstm.eval()
    y_pred = lstm(X_test)
    loss = criterion(y_pred, y_test)
    predicted_labels = y_pred.round()
    accuracy = (predicted_labels == y_test).sum().item() / y_test.size(0)
    print('Validation/Test Loss: {:.4f}, Accuracy: {:.4f}'.format(loss.item(), accuracy))

# %%
