import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import os, itertools, librosa
import numpy as np
import matplotlib.pylab as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.models import load_model

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues, name='test'):
    plt.figure(figsize=(15, 15))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    trained_classes = classes
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(name + title, fontsize=15)
    tick_marks = np.arange(len(classes))
    plt.xticks(np.arange(len(trained_classes)), classes, rotation=90, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.round(cm[i, j], 2), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=15)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.savefig(f"{name} confusion_matrix.png", dpi=300)
    return cm

x, label, classes = [], [], os.listdir('dataset')
for idx, i in enumerate(os.listdir('dataset')):
    for j in os.listdir(f'dataset/{i}'):
        avi_path = f'dataset/{i}/{j}'
        
        y, sr = librosa.load(avi_path)
        if y.shape[0] < sr:#padding zero
            zero_padding = np.zeros(sr - y.shape[0], dtype=np.float32)
            y = np.concatenate([y, zero_padding], axis=0)
        else:
            y = y[0:sr]
        melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr) # 梅尔频谱
        mfcc = librosa.feature.mfcc(y=y, sr=sr) # 梅尔倒谱
        
        if melspectrogram.shape != (128, 44) and mfcc.shape != (20, 44):
            print(avi_path)
        else:
            x.append(np.concatenate([melspectrogram, mfcc], axis=0))
            label.append(idx)

x, y = np.expand_dims(np.stack(x), axis=-1), np.array(label)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=0, shuffle=True)

model = load_model('model.h5')

y_pred = np.argmax(model.predict(x_val, verbose=0), axis=-1)
print('Val Set:')
print(classification_report(y_val, y_pred,target_names=classes))
cm = confusion_matrix(y_val, y_pred)
plot_confusion_matrix(cm, classes=classes, name='val')

y_pred = np.argmax(model.predict(x_test, verbose=0), axis=-1)
print('Test Set:')
print(classification_report(y_test, y_pred,target_names=classes))
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=classes, name='test')