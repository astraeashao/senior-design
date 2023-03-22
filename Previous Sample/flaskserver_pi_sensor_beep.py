import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
from flask import Flask,request,render_template
from flask_cors import CORS
import json
import os, librosa
import numpy as np
import tflite_runtime.interpreter as tflite

import pyaudio
import wave
import time
import threading

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
CORS(app, resources=r'/*')
classes = ['ambient and wind', 'baby cry', 'bathroom', 'buzzing', 'door', 'rain', 'silence', 'siren', 'snoring', 'street noise', 'talking']
tflite_model = tflite.Interpreter(model_path='model-quantized.tflite')
tfstate = False #audio handle state: True is run
threshold = 1024 #sensor threshold
sensorstate = 0
sensorstate = False
buzzerstate = False
sensor_PIN = 4 #BCM
buzzer_PIN = 23 #BCM
nowstate = 0   #classifier result: 1 is snoring 0 is other noise
laststate = 0  #last classifier result
audiocount = 1 #audio wavfile name encode
audiolimit = 10#encode limit, when meet it,turn to 1
audiolist = [0  for x in range(0,audiolimit)] #tflite classifier wav handle queue 
time_count = 0 #record count
t1 = None      #threading 1
t2 = None      #threading 2
def sensor_callback(channel):
    global sensorstate,sensor_PIN
    if GPIO.input(sensor_PIN):     # if == 1  
        print("Rising edge detected")
        sensorstate = False       # Sensor Trig is LOW
    else:                          # if != 1  
        print("Falling edge detected")
        sensorstate = True
def audio_handle(record_second):
    global time_count,audiocount,audiolimit,tfstate,audiolist
    while True:
        if tfstate == False:
            pass
        elif tfstate == True:
            wave_out_path = "myvoice/audio"+str(audiocount)+".wav"
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 2
            RATE = 22050 #match dataset RATE
            time_count = 0
            p = pyaudio.PyAudio()
            wf = wave.open(wave_out_path, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            def callback(in_data, frame_count, time_info, status):
                wf.writeframes(in_data)
                if(time_count < 1):
                    return (in_data, pyaudio.paContinue)
                else:
                    return (in_data, pyaudio.paComplete)
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK,
                            stream_callback=callback)
            stream.start_stream()
            print("* recording")
            while stream.is_active():
                time.sleep(1.5)
                time_count += 1
            stream.stop_stream()
            stream.close()
            wf.close()
            p.terminate()
            print("* recording done!")
            try:
                index = audiolist.index(0)#find zero 
                audiolist[index] = audiocount
                audiocount = audiocount + 1
                if audiocount > 10:
                    audiocount = 1
            except Exception as e:
                pass
                
def tflite_predict():
    global tfstate,audiolist,nowstate,laststate,tflite_model
    tflite_model.allocate_tensors()
    tflite_input_details = tflite_model.get_input_details()#need (1,148,44,1)
    tflite_output_details = tflite_model.get_output_details()
    while True:
        if tfstate == False:
            pass
        elif tfstate == True:
            if audiolist[0] != 0:
                try:
                    avi_path = "myvoice/audio"+str(audiolist[0])+".wav"
                    y, sr = librosa.load(avi_path)
                    if y.shape[0] < sr:#padding zero
                        print("hello")
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
                        laststate = nowstate
                        nowstate = 1
                    else:
                        pred_label = 'other noise'
                        laststate = nowstate
                        nowstate = 0
                    audiolist = audiolist[1:] + [0] #queue
                    print(f'Path:{avi_path} pred_label:{pred_label}')
                except Exception as e:
                    pass
            
@app.route('/', methods=['get', 'post'])
def hello_world():
    return render_template("index.html")
@app.route('/tfstate', methods=['get', 'post'])
def setstate():
    global tfstate
    if request.method == "POST":
        msg = request.json.get('tfstate')
        if msg == True:
            tfstate = True
        elif msg == False:
            tfstate = False
        res = {"state": "OK","tfstate":str(tfstate)}
        print(res)
        return json.dumps(res)
    else:
        res = {"state": "ERROR"}
        return json.dumps(res)
@app.route('/state', methods=['get', 'post'])
def getstate():
    if request.method == "GET":
        res = {"state": "OK","nowstate": nowstate,"laststate": laststate,'sensorstate':sensorstate}
        print(res)
        return json.dumps(res)
    else:
        res = {"state": "ERROR"}
        return json.dumps(res)
@app.route('/threshold', methods=['get', 'post'])
def setthreshold():
    global threshold
    if request.method == "POST":
        msg = request.json.get('threshold')
        threshold = int(msg)
        res = {"state": "OK","threshold":str(threshold)}
        print(res)
        return json.dumps(res)
    else:
        res = {"state": "ERROR"}
        return json.dumps(res)
@app.route('/alarm', methods=['get', 'post'])
def Alarm():
    global buzzerstate,buzzer_PIN
    if buzzerstate == False:
        buzzerstate = True
        GPIO.output(buzzer_PIN,GPIO.HIGH)
    res = {"state": "OK","buzzstate":"Beep"}
    print(res)
    return json.dumps(res)
@app.route('/dealarm', methods=['get', 'post'])
def deAlarm():
    global buzzerstate,buzzer_PIN
    if buzzerstate == True:
        buzzerstate = False
        GPIO.output(buzzer_PIN,GPIO.LOW)
    res = {"state": "OK","buzzstate":"No Beep"}
    print(res)
    return json.dumps(res)
if __name__ == '__main__':
    t1 = threading.Thread(target=audio_handle, args=("1",))
    t1.damon = True
    t2 = threading.Thread(target=tflite_predict)
    t2.damon = True
    t1.start()
    t2.start()
    GPIO.setmode(GPIO.BCM) # Use physical pin numbering
    GPIO.setup(sensor_PIN, GPIO.IN)
    GPIO.setup(buzzer_PIN, GPIO.OUT)
    GPIO.add_event_detect(sensor_PIN,GPIO.BOTH,callback=sensor_callback) # Setup event both falling/rising edge
    app.run(host='0.0.0.0',port=5000,debug=False,threaded=True)
    


    
    
        