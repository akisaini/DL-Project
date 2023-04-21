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

# %%
emotion = []
intensity = []
file_path = []
act_gender = []
main_path = []

# %%
import glob

rootdir = os.getcwd() + os.path.sep + 'ravdess'
# print(rootdir)

# getting gender, emotion, actor(M/F)
for path in glob.glob(f'{rootdir}/*/**'):
    main_path.append(path)

    path_short = path.split(os.path.sep)[
        -1]  # getting all the audio filenames to create gender specific tabular audio data.

    file_path.append(path_short)

    info = path_short.split('-')
    emotion.append(int(info[2]) - 1)
    intensity.append(info[3])
    gender = int(info[-1].split('.')[0])
    act_gender.append('F' if gender % 2 == 0 else 'M')

audio_tab = pd.DataFrame.from_dict({
    'gender': act_gender,
    'emotion': emotion,
    'intensity': intensity,
    'path': file_path,
})
# audio_tab.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)

# %%
# get the mel-scaled spectrogram (ransform both the y-axis (frequency) to log scale,
# and the “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes.)
log_spectrogram = []

for item in main_path:
    y, sample_rate = librosa.load(item, duration=3, sr=44100,
                                  offset=0.5)  # Load an audio file as a floating point time series.
    spec = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=128, fmax=8000)
    db_spec = librosa.power_to_db(spec)  # Convert a power spectrogram (amplitude squared) to decibel (dB) units
    # temporally average spectrogram
    log_spectrogram.append(np.mean(db_spec, axis=0))

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
    print()

# Convert the padded list of lists into a NumPy array
padded_array = np.array(padded_list_of_lists)

# %%
# ohe of target variable using label encoder:
lb = LabelEncoder()
# y_all = to_categorical(lb.fit_transform(np.array(emotion)))
y_all = emotion

# splitting into training and testing:
X_train, X_test, y_train, y_test = train_test_split(padded_array, y_all, test_size=0.15, random_state=0)

# %%
# Normalizing the data
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# %%
# Reshape data to include 3D tensor
# Reshape and convert the input array to a PyTorch tensor
X_train = torch.tensor(X_train, dtype=torch.float).reshape(1224, 5, 1, -1)
X_test = torch.tensor(X_test, dtype=torch.float).reshape(216, 5, 1, -1)
# X_train = torch.tensor(X_train, dtype=torch.float)
# X_test = torch.tensor(X_test, dtype=torch.float)

y_train = torch.tensor(y_train, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)
print('get data and ready to train:', X_train.shape, X_test.shape)

# %%
# Define the training parameters
input_size = X_train.shape[3]
hidden_size = 96
# n_categories = y_train.shape[1]
n_categories = 8


class RNN(nn.Module):
    # implement RNN from scratch rather than using nn.RNN
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


rnn = RNN(input_size, hidden_size, n_categories)
print('the original model')

criterion = nn.NLLLoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

max_epochs = 50


def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    # print('bbbbbb')
    # print(output.shape, hidden.shape)
    # print(output, hidden)
    # print('aaaaaaaa')
    loss = criterion(output, category_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()


# %%
def predict(line_tensor):
    with torch.no_grad():
        hidden = rnn.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        return output


# Train the RNN model
for epoch in range(1, max_epochs + 1):
    current_loss = 0
    for i in range(X_train.size(0)):
        category_tensor = y_train[i].reshape(1).type(torch.LongTensor)
        line_tensor = X_train[i]
        output, loss = train(line_tensor, category_tensor)

        current_loss += loss

    pred = []
    for i in range(X_test.shape[0]):
        output = predict(X_test[i])
        output_item = torch.argmax(output).item()
        pred.append(output_item)

    print('epoch', epoch, 'loss', current_loss / i, 'f1_score', f1_score(pred, y_test, average='micro'))
    #     hidden = rnn.initHidden()   # torch.Size([1, n_hidden])
    #
    #     optimizer.zero_grad()
    #
    #     # Pass the entire sequence through the RNN
    #     for j in range(X_train.size(1)):
    #         output, hidden = rnn(X_train[i][j].unsqueeze(0), hidden)
    #
    #     # Compute loss on the final output only
    #     err = criterion(output[-1], y_train[i])
    #     loss += err.item()
    #     err.backward()
    #     optimizer.step()
    #
    # print(f"Epoch {epoch} loss: {loss/X_train.size(0)}")

exit()

# Epoch 30 loss: 0.355356015751954
total = 0
correct = 0

pred = []
for i in range(X_test.shape[0]):
    output = predict(X_test[i])
    output_item = torch.argmax(output).item()
    pred.append(output_item)

    if int(output_item) == int(y_test[i]):
        correct += 1
    total += 1
    #
    # with torch.no_grad():
    #     category_tensor = y_test[i]
    #     line_tensor = X_test[i]
    #
    #     hidden = rnn.init_hidden()
    #
    #     for i in range(line_tensor.size()[0]):
    #         output, hidden = rnn(line_tensor[i], hidden)
    #
    #     pred = (output[-1] >= 0.5)
    #
    #     # Update the variables for accuracy calculation
    #     total += 1
    #     correct += torch.all(torch.eq(pred, y_test[i]))

# Calculate the accuracy
accuracy = correct / total
print(f"Validation set accuracy: {accuracy}")
print(pred)
# %%
exit()
# Set the model to evaluation`` mode
rnn.eval()

# Initialize variables for accuracy calculation
# Initialize variables for accuracy calculation
correct = 0
total = 0

# Iterate over the validation set
for i in range(X_test.shape[0]):
    hidden = rnn.initHidden()

    # Pass the entire sequence through the RNN
    for j in range(X_test.shape[1]):
        output, hidden = rnn(X_test[i][j].unsqueeze(0), hidden)

    # Get the predicted label
    pred = (output[-1] >= 0.5)

    # Update the variables for accuracy calculation
    total += 1
    correct += torch.all(torch.eq(pred, y_test[i]))

# Calculate the accuracy
accuracy = correct / total
print(f"Validation set accuracy: {accuracy}")

# %%
# Validation set accuracy: 0.018518518656492233