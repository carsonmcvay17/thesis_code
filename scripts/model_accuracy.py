# imports
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np

from sources.Networks import NN
from sources.data_utils import Utils


nn = NN()
utils = Utils()
ds_train, ds_test = utils.dataset()
model, activations = nn.trained_network()
history = model.fit(ds_train,
                    epochs=6,
                    validation_data=ds_test,)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()