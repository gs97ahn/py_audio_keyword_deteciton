# DNN

# Import Libraries
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Input, Dropout


def data_preprocessing(t):
    data_train = pd.DataFrame(np.load("./max_normalized/train_dataset_{0}.npy".format(str(t)), allow_pickle=True))
    data_test = pd.DataFrame(np.load("./max_normalized/test_dataset_{0}.npy".format(str(t)), allow_pickle=True))
    data_train.columns = ['feature', 'label']
    data_test.columns = ['feature', 'label']

    X_train = np.array(data_train.feature.tolist()) # train data
    Y_train = np.array(data_train.label.tolist()) # train data
    X_test = np.array(data_test.feature.tolist()) # test data
    Y_test = np.array(data_test.label.tolist()) # test label

    lb = LabelEncoder()
    Y_train = to_categorical(lb.fit_transform(Y_train))
    Y_test = to_categorical(lb.fit_transform(Y_test))

    return X_train, Y_train, X_test, Y_test


def dnn_model(X_train, Y_train, X_test, Y_test, t, n):
    if t == "mels":
        d1 = 8000
    else:
        d1 = 7800

    model = tf.keras.models.Sequential()

    model.add(Flatten())
    model.add(Dense(2048, input_shape=(d1,), activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='sgd')

    model.fit(X_train, Y_train, batch_size=64, epochs=72, validation_data=(X_test, Y_test),
              callbacks=[tf.keras.callbacks.ModelCheckpoint(
                  filepath=("./test/{0}_dnn_best_model_{1}.h5".format(str(t), str(n))),
                  save_weights_only=False, monitor='val_acc', save_best_only=True)])

    model.save("./test/{0}_dnn_model_{1}.h5".format(str(t), str(n)))

    print(model.summary())


if __name__ == "__main__":
    t = "mels"
    X_train, Y_train, X_test, Y_test = data_preprocessing(t)
    print("===============MEL-SPECTROGRAM===============")
    dnn_model(X_train, Y_train, X_test, Y_test, t, 0)
    # for i in range(14, 30):
    #     print("===== mels-model", i, " =====")
    #     dnn_model(X_train, Y_train, X_test, Y_test, t, i)

    t = "mfcc"
    X_train, Y_train, X_test, Y_test = data_preprocessing(t)
    print("===============MFCC===============")
    dnn_model(X_train, Y_train, X_test, Y_test, t, 0)
    # for i in range(30):
    #     print("===== mfcc-model", i, " =====")
    #     dnn_model(X_train, Y_train, X_test, Y_test, t, i)