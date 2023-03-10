import matplotlib.pyplot as plt
import pandas as pd

logs = pd.read_csv('train.log')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(logs['loss'], label='train')
plt.plot(logs['val_loss'], label='val')
plt.legend()
plt.title('loss')
plt.xlabel('epoch')

plt.subplot(1, 2, 2)
plt.plot(logs['accuracy'], label='train')
plt.plot(logs['val_accuracy'], label='val')
plt.legend()
plt.title('acc')
plt.xlabel('epoch')

plt.tight_layout()
plt.show()