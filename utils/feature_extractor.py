import os
import numpy as np
import librosa
from sklearn import preprocessing
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config.config import get_config

config = get_config()

class FeatureExtractor():
    def __init__(self):
        self.train_true_ctr_path = config.train_true_ctr_path
        self.train_false_ctr_path = config.train_false_ctr_path
        self.test_true_ctr_path = config.test_true_ctr_path
        self.test_false_ctr_path = config.test_false_ctr_path

        self.train_true_folder_path = config.train_true_folder_path
        self.train_false_folder_path = config.train_false_folder_path
        self.test_true_folder_path = config.test_true_folder_path
        self.test_false_folder_path = config.test_false_folder_path

        self.n_train_dataset = config.n_train_dataset
        self.n_test_dataset = config.n_test_dataset
        self.data = config.data
        self.n_fft = config.n_fft
        self.n_mels = config.n_mels
        self.n_mfcc = config.n_mfcc
        self.hop_length = config.hop_length
        self.win_length = config.win_length

    def create_folder(self):
        os.makedirs(config.dataset_folder_path, exist_ok=True)

    def retrieve_data(self, data_path):
        ctr = open(data_path, 'r')
        ctr_data = ctr.readlines()
        for data in ctr_data:
            ctr_data = ctr_data.replace('\n', '.wav')
        return ctr_data

    def create_np(self):
        train_dataset = np.zeros(shape=(self.n_train_dataset, 2), dtype=object)
        test_dataset = np.zeros(shape=(self.n_test_dataset, 2), dtype=object)
        return train_dataset, test_dataset

    def get_mels(self, folder_path, data_path):
        full_path = os.path.join(folder_path, data_path)
        try:
            X, sample_rate = librosa.load(full_path, res_type='kaiser_fast')
            mels = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=self.n_fft, n_mels=self.n_mels,
                                                  hop_length=self.hop_length, win_length=self.win_length).T
            log_mels = librosa.power_to_db(mels)
        except Exception:
            print('ERROR: Parsing file failed')
            log_mels = None

        # trim to get keyword speech and match shape
        m = log_mels.shape[0] // 2
        feature = log_mels[m - 100:m + 100]
        feature = np.abs(feature)
        feature = preprocessing.normalize(feature, norm='max')
        feature = feature.reshape(200, self.n_mels, 1)
        return feature

    def get_mfcc(self, folder_path, data_path):
        fullpath = os.path.join(folder_path, data_path)
        try:
            X, sample_rate = librosa.load(fullpath, res_type='kaiser_fast')
            mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_fft=self.n_fft, n_mfcc=self.n_mfcc,
                                        hop_length=self.hop_length, win_length=self.win_length)
        except Exception:
            print('ERROR: Parsing file failed')
            mfcc = None
        mfcc_delta1 = librosa.feature.delta(mfcc, width=3)
        mfcc_delta2 = librosa.feature.delta(mfcc, width=3, order=2)
        feature = np.concatenate([mfcc, mfcc_delta1, mfcc_delta2], axis=0).T

        # trim to get keyword speech and match shape
        m = feature.shape[0] // 2
        feature = feature[m - 100:m + 100]
        feature = np.abs(feature)
        feature = preprocessing.normalize(feature, norm='max')
        feature = feature.reshape(200, 39, 1)
        return feature

    def save_feature(self, data, path):
        np.save(path, data, allow_pickle=True)


if __name__ == "__main__":
    feature_extractor = FeatureExtractor()

    # create folder
    feature_extractor.create_folder()

    # train data
    train_true_data = feature_extractor.retrieve_data(feature_extractor.train_true_ctr_path)
    train_false_data = feature_extractor.retrieve_data(feature_extractor.train_false_ctr_path)

    # test data
    test_true_data = feature_extractor.retrieve_data(feature_extractor.test_true_ctr_path)
    test_false_data = feature_extractor.retrieve_data(feature_extractor.test_false_ctr_path)

    # create numpy array
    mels_train_data, mels_test_data = feature_extractor.create_np()
    mfcc_train_data, mfcc_test_data = feature_extractor.create_np()

    # mels feature extraction for training data
    for i in range(0, mels_train_data.shape[0], 2):
        # true data
        mels_train_data[i, 0] = feature_extractor.get_mels(feature_extractor.train_true_ctr_path, train_true_data[i])
        mels_train_data[i, 1] = 'true'
        print('Mels Train Data:', i, '/', mels_train_data.shape[0])
        # false data
        mels_train_data[i + 1, 0] = feature_extractor.get_mels(feature_extractor.train_false_ctr_path, train_false_data[i])
        mels_train_data[i + 1, 1] = 'false'
        print('Mels Train Data:', i + 1, '/', mels_train_data.shape[0])
    feature_extractor.save_feature(mels_train_data, config.train_dataset_mels_path)

    # mels feature extraction for testing data
    for i in range(0, mels_test_data.shape[0], 2):
        # true data
        mels_test_data[i, 0] = feature_extractor.get_mels(feature_extractor.test_true_ctr_path, test_true_data[i])
        mels_test_data[i, 1] = 'true'
        print('Mels Test Data:', i, '/', mels_test_data.shape[0])
        # false data
        mels_test_data[i + 1, 0] = feature_extractor.get_mels(feature_extractor.test_false_ctr_path, test_false_data[i])
        mels_test_data[i + 1, 1] = 'false'
        print('Mels Test Data:', i + 1, '/', mels_test_data.shape[0])
    feature_extractor.save_feature(mels_test_data, config.test_dataset_mels_path)

    # mfcc feature extraction for training data
    for i in range(0, mfcc_train_data.shape[0], 2):
        # true data
        mfcc_train_data[i, 0] = feature_extractor.get_mfcc(feature_extractor.train_true_ctr_path, train_true_data[i])
        mfcc_train_data[i, 1] = 'true'
        print('MFCC Train Data:', i, '/', mfcc_train_data.shape[0])
        # false data
        mfcc_train_data[i + 1, 0] = feature_extractor.get_mfcc(feature_extractor.train_false_ctr_path, train_false_data[i])
        mfcc_train_data[i + 1, 1] = 'false'
        print('MFCC Train Data:', i + 1, '/', mfcc_train_data.shape[0])
    feature_extractor.save_feature(mfcc_train_data, config.train_dataset_mfcc_path)

    # mfcc feature extraction for testing data
    for i in range(0, mfcc_test_data.shape[0], 2):
        # true data
        mfcc_test_data[i, 0] = feature_extractor.get_mfcc(feature_extractor.test_true_ctr_path, test_true_data[i])
        mfcc_test_data[i, 1] = 'true'
        print('MFCC Train Data:', i, '/', mfcc_test_data.shape[0])
        # false data
        mfcc_test_data[i + 1, 0] = feature_extractor.get_mfcc(feature_extractor.test_false_ctr_path, test_false_data[i])
        mfcc_test_data[i + 1, 1] = 'false'
        print('MFCC Test Data:', i + 1, '/', mfcc_test_data.shape[0])
        feature_extractor.save_feature(mfcc_test_data, config.test_dataset_mfcc_path)