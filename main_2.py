#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa
from IPython.display import Audio #helps play sound
import torch
import os
import sys
import keras
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential 
from keras.regularizers import l2
from keras.layers import  Input, Flatten, Dropout, Activation, BatchNormalization, Dense
from keras.optimizer_experimental import sgd
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
#%%
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
    print(splt[-1]) #getting all the audio filenames to create gender specific tabular audio data. 
    file_path.append(splt[-1])
#%%   
#getting gender, emotion, actor(M/F) 
for i in range(len(file_path)):
    info = file_path[i].split('-')
    #print(info)
    emotion.append(info[2])
    intensity.append(info[3])
    gender.append(info[-1])
    
# %%
act_gender = []
for i in gender:
    gend = i.split('.')
    act_gender.append(gend[0])
    
#converting list elements from str to int
act_gender = list(map(int, act_gender))
act_gender = ['F' if i%2==0 else 'M' for i in act_gender]
# %%
audio_tab = pd.DataFrame(emotion)
# %%
audio_tab.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
audio_tab = pd.concat([pd.DataFrame(act_gender), audio_tab, pd.DataFrame(intensity), pd.DataFrame(file_path)], axis = 1)
audio_tab.columns = ['gender', 'emotion', 'intensity', 'path']
# %%
#EXTRACTING LOG MEL SPECTROGRAM MEAN VALUES
df = pd.DataFrame(columns=['mel_spectrogram'])
counter=0
main_path = []
for i in glob.glob(f'{rootdir}/*/**'):
    main_path.append(i)

#%%
num = []
sr_lst = []
for i in range(len(main_path)):
    X, sample_rate = librosa.load(main_path[i],duration=3,sr=38500)
    num.append(X)
    sr_lst.append(sample_rate)
    #get the mel-scaled spectrogram (ransform both the y-axis (frequency) to log scale, and the “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes.)
#%%
log_spectrogram = []
for i in range(len(num)):
    spec = librosa.feature.melspectrogram(y=num[i], sr=sr_lst[i], n_mels=128,fmax=8000) 
    db_spec = librosa.power_to_db(spec)
    #temporally average spectrogram
    log_spectrogram.append(np.mean(db_spec, axis = 0))
        
df['mel_spectrogram'] = log_spectrogram

df_main = pd.concat([audio_tab, df], axis = 1)

#%%
split_df = pd.DataFrame(df_main['mel_spectrogram'].tolist())
split_df = split_df.iloc[:,:196]

#%%
df_main = pd.concat([audio_tab, split_df], axis = 1)
# %%
#splitting into training and testing:
train_data, test_data = train_test_split(df_main, test_size=0.15, random_state=0)
#%%
X_train = train_data.iloc[:, 4:]
y_train = train_data.loc[:,'emotion'] #train target label

X_test = test_data.iloc[:, 4:]
y_test = test_data.loc[:, 'emotion'] #test target label

#%%
#Normalizing the data 
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

# %%
#converting into np array:
X_train = np.array(X_train, dtype = np.float32)
y_train = np.array(y_train, dtype = np.float32)
X_test = np.array(X_test, dtype = np.float32)
y_test = np.array(y_test, dtype = np.float32)

# %%
#ohe of target variable using label encoder:
lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train))
y_test = to_categorical(lb.fit_transform(y_test))
# %%
X_train = X_train.reshape(1224, 14, 14, 1)
X_test = X_test.reshape(216, 14, 14, 1)

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
#%%
# Define the 2D CNN architecture
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(14, 14, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

#model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
#model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))
#%%
# Compile the model with an appropriate optimizer, loss function and metrics
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.0001))
# %%
checkpoint = ModelCheckpoint("best_initial_model.hdf5", monitor='val_accuracy', verbose = 1, save_best_only = True, mode='max')

model_history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data = (X_test, y_test), callbacks = [checkpoint])

# %%
model.evaluate(X_test, y_test)