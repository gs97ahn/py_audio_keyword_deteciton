import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

from config.config import get_config

config = get_config()


class CNN():
    def __init__(self):
        self.data = config.data
        self.filters = config.cnn_filters
        self.pool_size = config.cnn_pool_size
        self.strides = config.strides
        self.dense1 = config.cnn_dense1
        self.dense2 = config.cnn_dense2
        self.dense3 = config.cnn_dense3
        self.dropout_rate = config.cnn_dropout_rate
        self.model_folder_path = config.model_folder_path

        if self.data == 'mels':
            self.train_dataset_path = config.train_dataset_mels_path
            self.test_dataset_path = config.test_dataset_mels_path
            self.input_shape = [config.mels_input_2d1, config.mels_input_2d2, config.mels_input_2d3]
            self.best_model_path = self.model_folder_path + 'best_cnn_mels.h5'
            self.model_path = self.model_folder_path + 'cnn_mels.h5'

        elif self.data == 'mfcc':
            self.train_dataset_path = config.train_dataset_mfcc_path
            self.test_dataset_path = config.test_dataset_mfcc_path
            self.input_shape = [config.mfcc_input_2d1, config.mfcc_input_2d2, config.mels_input_2d3]
            self.best_model_path = self.model_folder_path + 'best_cnn_mfcc.h5'
            self.model_path = self.model_folder_path + 'cnn_mfcc.h5'

        else:
            print('Error: Incorrect configuration data inputted')
            exit(1)

    def load_dataset(self):
        train_data = pd.DataFrame(np.load(self.train_dataset_path), allow_pickle=True)
        test_data = pd.DataFrame(np.load(self.test_dataset_path), allow_pickle=True)
        train_data.columns = ['feature', 'label']
        test_data.columns = ['feature', 'label']

        x_train = np.array(train_data.feature.tolist())
        y_train = np.array(train_data.label.tolist())
        x_test = np.array(test_data.feature.tolist())
        y_test = np.array(test_data.label.tolist())

        lb = LabelEncoder()
        y_train = keras.utils.to_categorical(lb.fit_transform(y_train))
        y_test = keras.utils.to_categorical(lb.fit_transform(y_test))

        return x_train, y_train, x_test, y_test

    def model(self):
        input_layer = keras.Input(self.input_shape)
        conv2d1 = keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding='same',
                                      activation=tf.nn.relu)(input_layer)
        avgpool2d1 = keras.layers.AvgPool2D(pool_size=self.pool_size, strides=self.strides)(conv2d1)
        conv2d2 = keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding='same',
                                      activation=tf.nn.relu)(avgpool2d1)
        avgpool2d2 = keras.layers.AvgPool2D(pool_size=self.pool_size, strides=self.strides)(conv2d2)
        dropout1 = keras.layers.Dropout(rate=self.dropout_rate)(avgpool2d2)
        flatten = keras.layers.Flatten()(dropout1)
        dense1 = keras.layers.Dense(self.dense1, activation=tf.nn.relu)(flatten)
        dropout2 = keras.layers.Dropout(rate=self.dropout_rate)(dense1)
        dense2 = keras.layers.Dense(self.dense2, activation=tf.nn.relu)(dropout2)
        dropout3 = keras.layers.Dropout(rate=self.dropout_rate)(dense2)
        output_layer = keras.layers.Dense(self.dense3, activation=tf.nn.relu)(dropout3)

        return keras.Model(inputs=input_layer, outputs=output_layer)

    def train(self):
        x_train, y_train, x_test, y_test = self.load_dataset()
        model = self.model()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  callbacks=[tf.keras.callbacks.ModelCheckpoint(
                      filepath=self.best_model_path, save_weights_only=False, monitor='val_acc', save_best_only=True)])
        model.save(self.model_path)