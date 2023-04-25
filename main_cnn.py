#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa
from IPython.display import Audio #helps play sound
import os
import sys
import keras
import soundfile as sf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential 
from keras.regularizers import l2
from keras.layers import  Input, Flatten, Dropout, Activation, BatchNormalization, Dense
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
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
path = os.getcwd()

rootdir = path+os.path.sep+'ravdess'

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
    X, sample_rate = librosa.load(main_path[i],duration=3,sr=44100,offset=0.5)
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

#The final tabular dataframe containing frequency in dB 
df_main = pd.concat([audio_tab, split_df], axis = 1)

#%%
#Imputing NaN's:
for i in range(0, 1440):
    df_main.iloc[i,4:] = df_main.iloc[i,4:].fillna(df_main.iloc[i,4:].median())
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

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
# %%
#ohe of target variable using label encoder:
lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train))
y_test = to_categorical(lb.fit_transform(y_test))
# %%
#Reshape data to include 3D tensor
X_train = X_train[:,:,np.newaxis]
X_test = X_test[:,:,np.newaxis]
#%%
'''
model = Sequential()
#64 filters
model.add(Conv1D(128, kernel_size=(10), activation='relu', input_shape=(X_train.shape[1],1))) #1st Conv layer
#128 filters
model.add(Conv1D(128, kernel_size=(10),activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))) #2nd Conv layer
model.add(MaxPooling1D(pool_size=(8)))#pooling layer
model.add(Dropout(0.4))
model.add(Conv1D(256, kernel_size=(10),activation='relu')) # 3rd Conv layer
model.add(MaxPooling1D(pool_size=(8))) #Another pooling layer
model.add(Dropout(0.4))
model.add(Flatten()) # finally flattened before dense layers
model.add(Dense(256, activation='elu')) # dense layer
model.add(Dropout(0.4))
model.add(Dense(8, activation='softmax')) # output layer 
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001))
model.summary()
'''
#Baseline CNN model: 

model=Sequential()
###first layer
model.add(Dense(100,input_shape=(259,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###final layer
model.add(Dense(8))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))

# %%
checkpoint = ModelCheckpoint("best_initial_model.hdf5", monitor='val_accuracy', verbose=1,
    save_best_only=True, mode='max')

model_history=model.fit(X_train, y_train,batch_size=64, epochs=1000, validation_data=(X_test, y_test),callbacks=[checkpoint])

#model.save('mymodel')
# %%
model.evaluate(X_test, y_test)
# %%
df_main
# %%
import pickle 
#%%
print("Creating Pickle File")
pickle.dump(model, open('ml_model.pkl', 'wb'))
# %%
model.evaluate(X_train, y_train)
# %%
