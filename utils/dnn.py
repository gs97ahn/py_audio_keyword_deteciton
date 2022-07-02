import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

from config.config import get_config

config = get_config()


class DNN():
    def __init__(self):
        self.data = config.data
        self.dense1 = config.dnn_dense1
        self.dense2 = config.dnn_dense2
        self.dense3 = config.dnn_dense3
        self.dense4 = config.dnn_dense4
        self.dense5 = config.dnn_dense5
        self.dense6 = config.dnn_dense6
        self.dense7 = config.dnn_dense7
        self.dropout_rate = config.dnn_dropout_rate
        self.model_folder_path = config.model_folder_path

        if self.data == 'mels':
            self.train_dataset_path = config.train_dataset_mels_path
            self.test_dataset_path = config.test_dataset_mels_path
            self.input_shape = [config.mels_input_1d]
            self.best_model_path = self.model_folder_path + 'best_dnn_mels.h5'
            self.model_path = self.model_folder_path + 'dnn_mels.h5'

        elif self.data == 'mfcc':
            self.train_dataset_path = config.train_dataset_mfcc_path
            self.test_dataset_path = config.test_dataset_mfcc_path
            self.input_shape = [config.mfcc_input_1d]
            self.best_model_path = self.model_folder_path + 'best_dnn_mfcc.h5'
            self.model_path = self.model_folder_path + 'dnn_mfcc.h5'

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

        y_train = keras.utils.to_categorical(LabelEncoder().fit_transform(y_train))
        y_test = keras.utils.to_categorical(LabelEncoder().fit_transform(y_test))

        return x_train, y_train, x_test, y_test

    def model(self):
        input_layer = keras.layers.Flatten()
        inputs = keras.Input(self.input_shape)(input_layer)
        dense1 = keras.layers.Dense(self.dense1, activation=tf.nn.relu)(inputs)
        dropout1 = keras.layers.Dropout(self.dropout_rate)(dense1)
        dense2 = keras.layers.Dense(self.dense2, activation=tf.nn.relu)(dropout1)
        dropout2 = keras.layers.Dropout(self.dropout_rate)(dense2)
        dense3 = keras.layers.Dense(self.dense3, activation=tf.nn.relu)(dropout2)
        dropout3 = keras.layers.Dropout(self.dropout_rate)(dense3)
        dense4 = keras.layers.Dense(self.dense4, activation=tf.nn.relu)(dropout3)
        dropout4 = keras.layers.Dropout(self.dropout_rate)(dense4)
        dense5 = keras.layers.Dense(self.dense5, activation=tf.nn.relu)(dropout4)
        dropout5 = keras.layers.Dropout(self.dropout_rate)(dense5)
        dense6 = keras.layers.Dense(self.dense6, activation=tf.nn.relu)(dropout5)
        dropout6 = keras.layers.Dropout(self.dropout_rate)(dense6)
        output_layer = keras.layers.Dense(self.dense7, activation=tf.nn.softmax)(dropout6)
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
