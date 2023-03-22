import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
from flask import Flask,request,render_template
from flask_cors import CORS
import json
import os, re,librosa
import numpy as np
import tflite_runtime.interpreter as tflite
from apscheduler.schedulers.background import BackgroundScheduler
import pyaudio
import wave
import time
import threading
import csv
import smtplib
from email import encoders
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.header import Header
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
DIR_PIN = 15
STEP_PIN = 14

nowstate = 0   #classifier result: 1 is snoring 0 is other noise
laststate = 0  #last classifier result
audiocount = 1 #audio wavfile name encode
audiolimit = 10#encode limit, when meet it,turn to 1
audiolist = [0  for x in range(0,audiolimit)] #tflite classifier wav handle queue 
time_count = 0 #record count
t1 = None      #audio
t2 = None      #predict
t3 = None      #motor
sched = BackgroundScheduler(timezone='MST')
job_id = "buzzer"
job_id2 = "snoring"

CW = 1 # 1 foreward
STEP_Action = 0 #when inequal 0, this val will -1 and motor start ,when 0,motor stop
STEP_Count = 0 #when step foreward count +1,reverse —1
STEP_Max = 200 #200 STEP = 360° When Count >= 200, CW = 0 and turn reverse
STEP_Min = 0 #0 STEP = 0°,when Count <= 0,CW = 1 and turn foreward
STEP_Blockflag = False #30 seconds block motor run,such as between twice snoring and through back url
STEP_Blocktime = 30 #30 seconds
STEP_Blocktimer = 0 #mark moment time,block cancel timer 

snoringcount = 0
writeflag = False
email_host = 'smtp.gmail.com'
email_port = 587
email_user = 'exiashao@gmail.com'
email_pwd = 'bsxqqkqrrosmmtwq'
email_sender = 'exiashao@gmail.com'
email_receiver = 'vonstillberg@gmail.com'
filename = "snoring.csv"

def buzzer_close():
    global buzzerstate,buzzer_PIN
    buzzerstate = False
    GPIO.output(buzzer_PIN,GPIO.LOW)
def sensor_callback(channel):
    global sensorstate,sensor_PIN,STEP_Action
    if GPIO.input(sensor_PIN):     # if == 1  
        print("Rising edge detected")
        sensorstate = True       # Sensor Trig is LOW
    else:                          # if != 1  
        print("Falling edge detected")
        sensorstate = False
        STEP_Action = 0 #clear motor step action
def motor_run():
    global DIR_PIN,STEP_PIN,CW,STEP_Action,STEP_Count,STEP_Max,STEP_Min,STEP_Blockflag,STEP_Blocktimer,STEP_Blocktime
    while True:
        if (STEP_Blockflag == True) and (int(time.time()) - STEP_Blocktimer >= STEP_Blocktime):
            STEP_Blockflag = False #Cancel block
        if STEP_Action > 0:
            GPIO.output(STEP_PIN,GPIO.HIGH)
            time.sleep(.005)
            GPIO.output(STEP_PIN,GPIO.LOW)
            time.sleep(.005)
            if CW == 1:
                STEP_Count = STEP_Count + 1
            elif CW == 0:
                STEP_Count = STEP_Count - 1
            if STEP_Count>= STEP_Max:
                CW = 0 #reverse
                time.sleep(1.0)
                GPIO.output(DIR_PIN,CW)
            elif STEP_Count<= STEP_Min:
                CW = 1 #foreward
                time.sleep(1.0)
                GPIO.output(DIR_PIN,CW)
            STEP_Action = STEP_Action - 1

def writedata():
    global filename,snoringcount
    nowtime = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    try:
        if os.path.exists(filename):
            with open(filename,'a',newline='') as file:
                writer=csv.writer(file)
                writer.writerow([nowtime,str(snoringcount)])
        else:
            with open(filename,'a',newline='') as file:
                writer=csv.writer(file)
                writer.writerow(['Time','Snoring'])
                writer.writerow([nowtime,str(snoringcount)])
        snoringcount = 0
        print("write")
    except Exception as e:
        pass
def checkemail(email):
    reg="\w+[@][a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)+"
    result=re.findall(reg,email)
    if result:
        return True
    else:
        return False
def send_email(receiver,content,file):
    global email_user,email_pwd,email_host,email_sender
    message = MIMEMultipart()
    message['From'] = Header(email_user)
    message['To'] = Header(receiver)
    subject = 'Snoring Statistics'
    message['Subject'] = Header(subject, 'utf-8')
    message.attach(MIMEText(content,_charset='utf-8'))
    
    attachment = MIMEBase('application', 'octet-stream') #'octet-stream': binary data 
    attachment.set_payload(open(file, 'rb').read()) 
    encoders.encode_base64(attachment) 
    attachment.add_header('Content-Disposition', 'attachment',filename=file)
    message.attach(attachment) 
    
    try:
        obj = smtplib.SMTP()
        obj.connect(email_host, 25)  # 25 为 SMTP 端口号
        obj.login(email_user, email_pwd)
        obj.sendmail(email_sender, receiver, message.as_string())
        obj.close()
        print("send success")
    except smtplib.SMTPException:
        print("error")

def audio_handle(record_second):
    global time_count,audiocount,audiolimit,tfstate,audiolist
    while True:
        if tfstate == False:
            pass
        elif tfstate == True and sensorstate == True:
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
    global tfstate,audiolist,nowstate,laststate,tflite_model,snoringcount,STEP_Action,STEP_Blockflag,STEP_Blocktimer,STEP_Blocktime
    tflite_model.allocate_tensors()
    tflite_input_details = tflite_model.get_input_details()#need (1,148,44,1)
    tflite_output_details = tflite_model.get_output_details()
    while True:
        if tfstate == False:
            pass
        elif tfstate == True and sensorstate == True:
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
                        snoringcount = snoringcount + 1
                        if STEP_Blockflag == False:
                            STEP_Blockflag = True
                            STEP_Blocktimer = (int(time.time()))
                            STEP_Action = 25
                    else:
                        pred_label = 'other noise'
                        laststate = nowstate
                        nowstate = 0
                    audiolist = audiolist[1:] + [0] #queue
                    print(f'Path:{avi_path} pred_label:{pred_label}')
                except Exception as e:
                    pass
def GPIO_Init():
    global sensor_PIN,buzzer_PIN,DIR_PIN,STEP_PIN,CW
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(sensor_PIN, GPIO.IN)
    GPIO.setup(buzzer_PIN, GPIO.OUT)
    GPIO.setup(DIR_PIN, GPIO.OUT)
    GPIO.setup(STEP_PIN, GPIO.OUT)
    GPIO.output(DIR_PIN, CW)
    GPIO.add_event_detect(sensor_PIN,GPIO.BOTH,callback=sensor_callback) # Setup event both falling/rising edge            
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
    global nowstate,laststate,sensorstate
    if request.method == "GET":
        if sensorstate == True:
            res = {"state": "OK","nowstate": nowstate,"laststate": laststate,'sensorstate':sensorstate}
        else:
            res = {"state": "OK","nowstate": "","laststate": "",'sensorstate':sensorstate}
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
    global buzzerstate,buzzer_PIN,sched,job_id
    if buzzerstate == False:
        buzzerstate = True
        GPIO.output(buzzer_PIN,GPIO.HIGH)
        sched.add_job(buzzer_close, 'interval', id=job_id, minutes=1)
        sched.start()
    res = {"state": "OK","buzzstate":"Beep"}
    print(res)
    return json.dumps(res)
@app.route('/dealarm', methods=['get', 'post'])
def deAlarm():
    global buzzerstate,buzzer_PIN,sched,job_id
    if buzzerstate == True:
        buzzerstate = False
        GPIO.output(buzzer_PIN,GPIO.LOW)
        sched.remove_job(job_id)
    res = {"state": "OK","buzzstate":"No Beep"}
    print(res)
    return json.dumps(res)
@app.route('/start',methods=['get', 'post'])
def startwrite():
    global writeflag,job_id2
    if request.method == "GET":
        if writeflag == False:
            sched.add_job(writedata, 'interval', id=job_id2, minutes=10)
            sched.start()
            writeflag = True
            res = {"state": "OK"}
        else:
            res = {"state": "ERROR"}
        return json.dumps(res)
    else:
        res = {"state": "ERROR"}
        return json.dumps(res)
    
@app.route('/send/<email>',methods=['get', 'post'])
def senddata(email):
    global filename,writeflag,job_id2
    if request.method == "GET":
        if checkemail(email) == True:
            print(email)
            email_receiver  = email
            send_email(email_receiver,"Snoring ",filename)
            if writeflag == True:
                sched.remove_job(job_id2)
                writeflag = False
            res = {"state": "OK","email":email_receiver}
        else:
            res = {"state": "ERROR","email":"email invaild"}
        return json.dumps(res)
    else:
        res = {"state": "ERROR"}
        return json.dumps(res)
@app.route('/back',methods=['get', 'post'])
def back():
    global CW,DIR_PIN,STEP_Action,STEP_Count,STEP_Blockflag,STEP_Blocktimer
    if request.method == "GET":
        CW = 0 # 1 foreward
        time.sleep(1.0)
        GPIO.output(DIR_PIN,CW)
        STEP_Blockflag = True
        STEP_Blocktimer = (int(time.time()))
        STEP_Action = STEP_Count
        print("back")
        res = {"state": "OK"}
        return json.dumps(res)
    else:
        res = {"state": "ERROR"}
        return json.dumps(res)
if __name__ == '__main__':
    GPIO_Init()
    t1 = threading.Thread(target=audio_handle, args=("1",))
    t1.damon = True
    t2 = threading.Thread(target=tflite_predict)
    t2.damon = True
    t3 = threading.Thread(target=motor_run)
    t3.damon = True
    t1.start()
    t2.start()
    t3.start()
    app.run(host='0.0.0.0',port=5000,debug=False,threaded=True)
    


    
    
        

    
    
        
