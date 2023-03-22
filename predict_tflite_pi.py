import os, librosa
import numpy as np
import tflite_runtime.interpreter as tflite
classes = ['snoring', 'other noise']
print(classes)
quantized = True #True is test tflite(quantized),False is test tflite(no quantized)
if quantized == True:
    tflite_model = tflite.Interpreter(model_path='model.tflite')
else:
    tflite_model = tflite.Interpreter(model_path='model-quantized.tflite')
tflite_model.allocate_tensors()
tflite_input_details = tflite_model.get_input_details()#need (1,148,44,1)
tflite_output_details = tflite_model.get_output_details()

while True:
    avi_path = input('Input Voice Path:')
    
    y, sr = librosa.load(avi_path)
    if y.shape[0] < sr:#padding zero
        zero_padding = np.zeros(sr - y.shape[0], dtype=np.float32)
        y = np.concatenate([y, zero_padding], axis=0)
    else:
        y = y[0:sr]
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr) # Melspect
    mfcc = librosa.feature.mfcc(y=y, sr=sr) # MFCC features
    x_tflite = np.expand_dims(np.concatenate([melspectrogram, mfcc], axis=0), axis=0)#(1,148,44)->(1,148,44,1)
    x_tflite = np.expand_dims(x_tflite, axis=-1)
    tflite_model.set_tensor(tflite_input_details[0]['index'], x_tflite)  
    tflite_model.invoke()
    y_tflite = tflite_model.get_tensor(tflite_output_details[0]['index'])
    pred = np.argmax(y_tflite)
    if classes[pred] == 'snoring':
        pred_label = 'snoring'
    else:
        pred_label = 'other noise'
    print(f'Path:{avi_path} pred_label:{pred_label}')

