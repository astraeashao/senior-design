This is the main program used to predict and turn the motor,
By running flaskserver_pi_V8, 
then opening up the browser window and enter the ip address of the rasp pi in the webpage
Users can use the entire function of the in-pillow system: snoring detection, activate alarm, deactivate alarm, motor roll back,
sending email

ex: 192.168.99.61:5000/enable

activate alarm: /alarm
deactivate: /dealarm
roll back to original position: /back
start snoring count: /start
end and send snoring count: /send/xxxxx@xxx.com