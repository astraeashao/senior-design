import os, librosa
import numpy as np
import tensorflow as tf
from keras.models import load_model

model = load_model('model.h5')
converter =tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open('model.tflite', 'wb').write(tflite_model)
#quantize tflite model, default input type is float32
converter =tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open('model-quantized.tflite', 'wb').write(tflite_model)
