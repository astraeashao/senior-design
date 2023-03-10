import csv
import time
import pyaudio
import wave
import os, librosa
import numpy as np
#import tflite_runtime.interpreter as tflite
import tensorflow as tf
import matplotlib.pyplot as plt 
classes = ['snoring', 'other noise']
print(classes)
Method = 1  #1 is 10s per, 2 is 20s per, 3 is 30s per,4 is 60s per
Methodnums = 4
StoreFolder = ["Method1","Method2","Method3","Method4"]
Totalresult = []
Totalcompareresult = []
Totalacc = []
Totalresult_quantize = []
Totalcompareresult_quantize = []
Totalacc_quantize = []
result = []
compareresult = []
label60 = []
with open("label.csv") as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:
        label60.append(row[1])
print(label60)
restart = True # Use true if predict after record, False if analyze existing wavs

def RecordTestBlock(method,folder):
    count = 0 #cycle count
    audiocount = 0
    if method == 1:
        count = 6
        record_seconds = 10
    elif method == 2:
        count = 3
        record_seconds = 20
    elif method == 3:
        count = 2
        record_seconds = 30
    elif method == 4:
        count = 1
        record_seconds = 60
    for i in range(count):
        wave_out_path = folder + "/original" + str(audiocount+1)+".wav"
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 22050
        time_count = 0
        p = pyaudio.PyAudio()
        wf = wave.open(wave_out_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True)
        print("* recording")
        wav_data = stream.read(int(RATE * record_seconds))
        wf.writeframes(wav_data)
        stream.stop_stream()
        stream.close()
        wf.close()
        p.terminate()
        audiocount = audiocount + 1
        print("* recording done!")
def tflite_predict(method,folder,labal60):
    global classes,tflite_model,compareresult,result
    count = 0 #cycle count
    result = []
    compareresult = []
    if method == 1:
        count = 6
        record_seconds = 10
    elif method == 2:
        count = 3
        record_seconds = 20
    elif method == 3:
        count = 2
        record_seconds = 30
    elif method == 4:
        count = 1
        record_seconds = 60
    for i in range(count):
        avi_path = folder + "/original" + str(i+1)+".wav"
        y, sr = librosa.load(avi_path)
        if y.shape[0] < sr*record_seconds:
            zero_padding = np.zeros(sr*record_seconds - y.shape[0], dtype=np.float32)
            y = np.concatenate([y, zero_padding], axis=0)
        for j in range(record_seconds):
            y_temp = y[0+sr*j:sr+sr*j]
            print(y_temp.shape)
            melspectrogram = librosa.feature.melspectrogram(y=y_temp, sr=sr) 
            mfcc = librosa.feature.mfcc(y=y_temp, sr=sr) 
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
            result.append(pred_label)
            if pred_label == label60[j]:
                compare = 1
            else:
                compare = 0
            compareresult.append(compare)
            print(f'Path:{avi_path} seconds:{j+1} pred_label:{pred_label} compare:{compare}')
if restart == True:
    for i in range(Methodnums):
        print(StoreFolder[i])
        if not os.path.exists(StoreFolder[i]):
            os.makedirs(StoreFolder[i])
        RecordTestBlock(i+1,StoreFolder[i])
print("Start No Quantize Predict Test")
tflite_model = tf.lite.Interpreter(model_path='model.tflite')
#tflite_model = tflite.Interpreter(model_path='model.tflite')
tflite_model.allocate_tensors()
tflite_input_details = tflite_model.get_input_details()#need (1,148,44,1)
tflite_output_details = tflite_model.get_output_details()
for i in range(Methodnums):
    print(StoreFolder[i])
    tflite_predict(i+1,StoreFolder[i],label60)
    acc = sum(compareresult)/60*100
    print(StoreFolder[i])
    print(compareresult)
    print(acc)
    Totalresult.append(result)
    Totalcompareresult.append(compareresult)
    Totalacc.append(acc)
print("Test Done")
print(Totalcompareresult)
print(Totalacc)
log = open("noquantizetest.txt",mode="w",encoding="utf-8")
print(label60,file = log)
print(Totalresult,file = log)
print(Totalcompareresult,file = log)
print(Totalacc,file = log)
log.close()
x = np.arange(Methodnums)
bar = plt.bar(x, Totalacc, color='tab:red')
ax = plt.gca()
tt = ['10s/per', '20s/per', '30s/per', '60s/per']
ax.set_xticks(x)
ax.set_xticklabels(tt, rotation = 45)  
ax.set_xlabel("Method")
ax.set_ylabel("Acc")    
ax.set_title("No Quantize acc")
ax.bar_label(bar,fmt='%g%%')
plt.savefig("No Quantize.png",dpi=300,bbox_inches='tight')

print("Start Quantize Predict Test")
#tflite_model = tflite.Interpreter(model_path='model-quantized.tflite')
tflite_model = tf.lite.Interpreter(model_path='model-quantized.tflite')
tflite_model.allocate_tensors()
tflite_input_details = tflite_model.get_input_details()#need (1,148,44,1)
tflite_output_details = tflite_model.get_output_details()
for i in range(Methodnums):
    print(StoreFolder[i])
    tflite_predict(i+1,StoreFolder[i],label60)
    acc = sum(compareresult)/60*100
    print(StoreFolder[i])
    print(compareresult)
    print(acc)
    Totalresult_quantize.append(result)
    Totalcompareresult_quantize.append(compareresult)
    Totalacc_quantize.append(acc)
    
print("Test Done")
print(Totalcompareresult_quantize)
print(Totalacc_quantize)
log = open("quantizetest.txt",mode="w",encoding="utf-8")
print(label60,file = log)
print(Totalresult_quantize,file = log)
print(Totalcompareresult_quantize,file = log)
print(Totalacc_quantize,file = log)
log.close()
x = np.arange(Methodnums)
bar = plt.bar(x, Totalacc_quantize, color='tab:red')
ax = plt.gca()
tt = ['10s/per', '20s/per', '30s/per', '60s/per']
ax.set_xticks(x)
ax.set_xticklabels(tt, rotation = 45)  
ax.set_xlabel("Method")
ax.set_ylabel("Acc")    
ax.set_title("Quantize acc")
ax.bar_label(bar,fmt='%g%%')
plt.savefig("Quantize.png",dpi=300,bbox_inches='tight')
    