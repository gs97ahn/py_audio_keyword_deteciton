import numpy as np
import pandas as pd
import os
import subprocess
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

from config.config import get_config

config = get_config()


def load_dataset():
    test_mels_data = pd.DataFrame(np.load(config.test_dataset_mels_path), allow_pickle=True)
    test_mels_data.columns = ['feature', 'label']
    test_mfcc_data = pd.DataFrame(np.load(config.test_dataset_mfcc_path), allow_pickle=True)
    test_mfcc_data.columns = ['feature', 'label']

    x_test_mels = np.array(test_mels_data.feature.tolist())
    y_test_mels = np.array(test_mels_data.label.tolist())
    x_test_mfcc = np.array(test_mfcc_data.feature.tolist())
    y_test_mfcc = np.array(test_mfcc_data.label.tolist())

    y_test_mels = keras.utils.to_categorical(LabelEncoder().fit_transform(y_test_mels))
    y_test_mfcc = keras.utils.to_categorical(LabelEncoder().fit_transform(y_test_mfcc))

    return x_test_mels, y_test_mels, x_test_mfcc, y_test_mfcc


if __name__ == "__main__":
    x_test_mels, y_test_mels, x_test_mfcc, y_test_mfcc = load_dataset()

    x_test, y_test = None, None
    data = config.data
    model = config.model
    loaded_model = None
    input_shape = None

    # input shape depending on model type and data feature type
    if model == 'dnn':
        if data == 'mels':
            input_shape = [config.mels_input_1d]
            loaded_model = tf.keras.models.load_model(config.model_folder_path + 'best_dnn_mels.h5')
        elif data == 'mfcc':
            input_shape = [config.mfcc_input_1d]
            loaded_model = tf.keras.models.load_model(config.model_folder_path + 'best_dnn_mfcc.h5')
        else:
            print('Error: Incorrect feature type inputted')
            exit(1)
    elif model == 'cnn':
        if data == 'mels':
            input_shape = [config.mels_input_2d1, config.mels_input_2d2, config.mels_input_2d3]
            loaded_model = tf.keras.models.load_model(config.model_folder_path + 'best_cnn_mels.h5')
            x_test, y_test = x_test_mels, y_test_mels
        elif data == 'mfcc':
            input_shape = [config.mfcc_input_2d1, config.mfcc_input_2d2, config.mels_input_2d3]
            loaded_model = tf.keras.models.load_model(config.model_folder_path + 'best_cnn_mfcc.h5')
            x_test, y_test = x_test_mfcc, y_test_mfcc
        else:
            print('Error: Incorrect feature type inputted')
            exit(1)
    elif model == 'resnet':
        if data == 'mels':
            input_shape = [config.mels_input_2d1, config.mels_input_2d2, config.mels_input_2d3]
            loaded_model = tf.keras.models.load_model(config.model_folder_path + 'best_resnet_mels.h5')
            x_test, y_test = x_test_mels, y_test_mels
        elif data == 'mfcc':
            input_shape = [config.mfcc_input_2d1, config.mfcc_input_2d2, config.mels_input_2d3]
            loaded_model = tf.keras.models.load_model(config.model_folder_path + 'best_resnet_mfcc.h5')
            x_test, y_test = x_test_mfcc, y_test_mfcc
        else:
            print('Error: Incorrect feature type inputted')
            exit(1)
    else:
        print('Error: Incorrect model type inputted')
        exit(1)

    # create true score in txt
    with open('./utils/score_true.txt', 'w') as f:
        for i in range(0, y_test.shape[0], 2):
            predicted = model.predict(x_test[i].reshape(input_shape))
            predicted_ratio = predicted[0][1]
            if predicted_ratio > 1.0:
                predicted_ratio = 1.0
            elif predicted_ratio < 0.00000001:
                predicted_ratio = 0.00000001
            f.write(str(predicted_ratio))
            f.write('\n')
            print(i, "score true:", str(predicted_ratio))
    f.close()

    # create false score in txt
    with open('score_false.txt', 'w') as f:
        for i in range(1, y_test.shape[0], 2):
            predicted = model.predict(x_test[i].reshape(input_shape))
            predicted_ratio = predicted[0][1]
            if ratio > 1.0:
                ratio = 1.0
            if ratio < 0.00000001:
                ratio = 0.00000001
            f.write(str(predicted_ratio))
            f.write('\n')
            print(i, "score false:", str(predicted_ratio))
    f.close()

    # Equal Error Rate
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, './utils/doit_eer.bat')
    subprocess.call(filename)
