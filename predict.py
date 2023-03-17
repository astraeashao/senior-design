import os, librosa
import numpy as np
from keras.models import load_model

model = load_model('model.h5')

classes = ['snoring', 'other noise']
while True:
    avi_path = input('Input Voice Path:')
    
    y, sr = librosa.load(avi_path)
    if y.shape[0] < sr:#padding zero
        zero_padding = np.zeros(sr - y.shape[0], dtype=np.float32)
        y = np.concatenate([y, zero_padding], axis=0)
    else:
        y = y[0:sr]
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr) 
    mfcc = librosa.feature.mfcc(y=y, sr=sr) 
    
    x = np.expand_dims(np.concatenate([melspectrogram, mfcc], axis=0), axis=0)
    print(x.shape)
    pred = np.argmax(model.predict(x, verbose=0))
    print(f'Path:{avi_path} pred_label:{classes[pred]}')
