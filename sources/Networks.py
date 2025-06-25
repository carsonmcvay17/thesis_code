# imports
import tensorflow as tf
import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np

class NN:
    """
    Creates trained and untrained networks
    """
    def trained_network(n: int, ds_train: np.ndarray, ds_test: np.ndarray)->tuple:
        """
        Creates a trained nueral network
        Args:
        n: int number of neurons in penultimate layer
        ds_train: train dataset
        ds_test: test dataset

        Returns:
        Model: the network
        Activations: array of activations of penultimate layer
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28,28), name = 'test'),
            tf.keras.layers.Dense(128, activation='relu', name='test2'),
            tf.keras.layers.Dense(n, activation='tanh', name='neurons'),
            tf.keras.layers.Dense(10)
        ])
        model.compile(
            optimizer = tf.keras.optimizers.Adam(0.001),
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics = ['acc']
        )

        model.fit(
            ds_train, 
            epochs = 6,
            validation_data = ds_test,
        )

        intermediate_output = tf.keras.Model(model.input,
                                             model.get_layer('neurons').output)
        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        activations = []
        for x in x_train:
            foo = tf.cast(x, tf.float32)/255
            activations.append(intermediate_output(foo[None,]).numpy())
        for x in x_test:
            foo = tf.cast(x, tf.float32)/255
            activations.append(intermediate_output(foo[None,].numpy()))
        return model, activations
    
    def untrained_network(self, n: int)->tuple:
        """
        Same stuff as above just not trained
        Args:
        n: int number of neurons in penultimate layer

        Returns:
        Model: the network
        Activations: array of activations of penultimate layer
        """
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28), name='test'),
                                            tf.keras.layers.Dense(128, activation='relu', name='test2'),
                                            tf.keras.layers.Dense(n, activation='tanh', name='neurons'),
                                            tf.keras.layers.Dense(10)
                                            ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['acc'],
        )

        intermediate_output = tf.keras.Model(model.input,
                                             model.get_layer('neurons').output)
        

    
        (x_train, y_train), (x_test,y_test) = mnist.load_data()
        activations = []
        for x in x_train:
            foo = tf.cast(x,tf.float32)/255
            activations.append(intermediate_output(foo[None,]).numpy())
        for x in x_test:
            foo = tf.cast(x,tf.float32)/255
            activations.append(intermediate_output(foo[None,]).numpy())
        return model, activations