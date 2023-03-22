# senior-design
previous sample folder has all the previous revisiions of the main program

main project is the main program for all the functions within the pillow system, run this to simulate the project demo

modelrecord test is the program run to record a 4 minutes audio, then run prediction on each second

plot curve.py should be used with the train.log file generated from google collab, re-edit the file so that it reads in different csv

metrice.py produce confusion matrix, would need to edit train.log file name as well

predict.py and predict_tflite_pi.py is one running on normal python, one is converted to tflite running on rasp pi, user enter the path of audio wav want to test out
and it will predict it as snoring or other noise

modeldatatest.py requires downloading the sample data from google collab, create and put them in 2 folder: snoring    othernoise;
edit the program following comments, sample program is testing on 80 samples not included in training and validation
