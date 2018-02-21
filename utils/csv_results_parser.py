from numpy import genfromtxt
import matplotlib.pyplot as plt
import os

os.chdir('..')

data = genfromtxt('results/results_cnn1.csv',
                  delimiter=',',
                  names=['epoch',
                         'train_c_loss',
                         'train_c_accuracy',
                         'val_c_loss',
                         'val_c_accuracy',
                         'test_c_loss',
                         'test_c_accuracy'])

plt.figure(1)
plt.subplot(211)
plt.plot(data['epoch'], data['val_c_accuracy'])
plt.xlabel('Epoch')
#plt.ylabel('Validation Set Accuracy')
plt.title('Accuracy on the Validation Set')
plt.grid(True)
#plt.savefig('results_cnn1_loss.png')

plt.subplot(212)
plt.plot(data['epoch'], data['val_c_loss'])
plt.xlabel('Epoch')
#plt.ylabel('Validation Set Loss')
plt.title('Loss Value on the Validation Set')
plt.grid(True)
plt.tight_layout()
plt.savefig('results_cnn1_combined.png', bbox_inches='tight')
