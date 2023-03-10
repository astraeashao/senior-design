import os, librosa
import numpy as np
#import tensorflow as tf
import tflite_runtime.interpreter as tflite
classes = ['snoring','other noise']
print(classes)
quantized = False #True is test tflite(quantized),False is test tflite(no quantized)
testflag = True #True if testing on false alarm rate of noise, False if testing on accuracy of snoring
if quantized == False:
    #tflite_model = tf.lite.Interpreter(model_path='model.tflite')
    tflite_model = tflite.Interpreter(model_path='model.tflite')
else:
    #tflite_model = tf.lite.Interpreter(model_path='model-quantized.tflite')
    tflite_model = tflite.Interpreter(model_path='model-quantized.tflite')
if testflag == True:#test snoring
    filename = "snoring/1_"
else:
    filename = "other noise/0_"
tflite_model.allocate_tensors()
tflite_input_details = tflite_model.get_input_details()#need (1,148,44,1)
tflite_output_details = tflite_model.get_output_details()
datalen = 80 #total length, change to 500 to full, 80 for last 80 test
audiocount = 420 #starting number for audio, change to 0 for full 500, 420 for last 80
audiolimit = 10#encode limit, when meet it,turn to 1
acccount = 0
while True:
    avi_path = filename+str(audiocount)+".wav"
    y, sr = librosa.load(avi_path)
    if y.shape[0] < sr:
        zero_padding = np.zeros(sr - y.shape[0], dtype=np.float32)
        y = np.concatenate([y, zero_padding], axis=0)
    else:
        y = y[0:sr]
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr) 
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    x_tflite = np.expand_dims(np.concatenate([melspectrogram, mfcc], axis=0), axis=0)#(1,148,44)->(1,148,44,1)
    x_tflite = np.expand_dims(x_tflite, axis=-1)
    tflite_model.set_tensor(tflite_input_details[0]['index'], x_tflite)  
    tflite_model.invoke()

    y_tflite = tflite_model.get_tensor(tflite_output_details[0]['index'])
    pred = np.argmax(y_tflite)
    if classes[pred] == 'snoring':
        pred_label = 'snoring'
        if testflag == True:
            acccount = acccount + 1
    else:
        pred_label = 'other noise'
        if testflag == False:
            acccount = acccount + 1
    print(f'Path:{avi_path} pred_label:{pred_label}')
    audiocount = audiocount + 1
    if audiocount > 499:#change total number if added
        break
        audiocount = 0
acc = acccount/datalen*100
print(acc)



