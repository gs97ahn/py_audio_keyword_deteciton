import argparse

parser = argparse.ArgumentParser()


def get_config():
    config, unparsed = parser.parse_known_args()
    return config


# Path
parser.add_argument('--train_true_ctr_path', type=str, default='./HiLG2/ctr/train_true.ctl')
parser.add_argument('--train_false_ctr_path', type=str, default='./HiLG2/ctr/train_false.ctl')
parser.add_argument('--test_true_ctr_path', type=str, default='./HiLG2/ctr/test_true.ctl')
parser.add_argument('--test_false_ctr_path', type=str, default='./HiLG2/ctr/test_false.ctl')

parser.add_argument('--train_true_folder_path', type=str, default='./HiLG2/raw16k/train_true/')
parser.add_argument('--train_false_folder_path', type=str, default='./HiLG2/raw16k/train_false/')
parser.add_argument('--test_true_folder_path', type=str, default='./HiLG2/raw16k/test_true/')
parser.add_argument('--test_false_folder_path', type=str, default='./HiLG2/raw16k/test_false/')

parser.add_argument('--dataset_folder_path', type=str, default='./dataset/')
parser.add_argument('--train_dataset_mels_path', type=str, default='./dataset/train_dataset_mels.npy',
                    help='Path to mels train dataset')
parser.add_argument('--test_dataset_mels_path', type=str, default='./dataset/test_dataset_mels.npy',
                    help='Path to mels test dataset')
parser.add_argument('--train_dataset_mfcc_path', type=str, default='./dataset/train_dataset_mfcc.npy',
                    help='Path to mfcc train dataset')
parser.add_argument('--test_dataset_mfcc_path', type=str, default='./dataset/test_dataset_mfcc.npy',
                    help='Path to mfcc test dataset')
parser.add_argument('--model_folder_path', type=str, default='./model/', help='Path for saving model')


# Feature Extraction
parser.add_argument('--data', type=str, default='mels', help='Choose type of data (mels/mfcc)')
parser.add_argument('--n_train_dataset', type=int, default=24012, help='Number of train dataset')
parser.add_argument('--n_test_dataset', type=int, default=31968, help='Number of test dataset')
parser.add_argument('--data', type=str, default='mels', help='Choose type of data (mels/mfcc)')
parser.add_argument('--n_fft', type=int, default=512, help='Number of FFT')
parser.add_argument('--n_mels', type=int, default=40, help="Number of Mels")
parser.add_argument('--n_mfcc', type=int, default=13, help="Number of MFCC")
parser.add_argument('--hop_length', type=int, default=160, help='Number of hop length')
parser.add_argument('--win_length', type=int, default=400, help="Number of win length")


# Model
parser.add_argument('--model', type=str, default='cnn', help='Choose type of model (cnn/dnn/resnet')
parser.add_argument('--mels_input_1d', type=int, default=8000, help='Input shape of mels for 1d')
parser.add_argument('--mfcc_input_1d', type=int, default=7800, help='Input shape of mfcc for 1d')
parser.add_argument('--mels_input_2d1', type=int, default=200, help='Input shape of 1d mels for 2d')
parser.add_argument('--mels_input_2d2', type=int, default=40, help='Input shape of 2d mels for 2d')
parser.add_argument('--mels_input_2d3', type=int, default=1, help='Input shape of 3d mels for 2d')
parser.add_argument('--mfcc_input_2d1', type=int, default=200, help='Input shape of 1d mfcc for 2d')
parser.add_argument('--mfcc_input_2d2', type=int, default=39, help='Input shape of 2d mfcc for 2d')
parser.add_argument('--mfcc_input_2d3', type=int, default=1, help='Input shape of 3d mfcc for 2d')


# DNN
parser.add_argument('--dnn_dense1', type=int, default=2048, help='Number of 1st dense unit for DNN')
parser.add_argument('--dnn_dense2', type=int, default=1024, help='Number of 2nd dense unit for DNN')
parser.add_argument('--dnn_dense3', type=int, default=512, help='Number of 3rd dense unit for DNN')
parser.add_argument('--dnn_dense4', type=int, default=256, help='Number of 4th dense unit for DNN')
parser.add_argument('--dnn_dense5', type=int, default=128, help='Number of 5th dense unit for DNN')
parser.add_argument('--dnn_dense6', type=int, default=64, help='Number of 6th dense unit for DNN')
parser.add_argument('--dnn_dense7', type=int, default=2, help='Number of 7th dense unit for DNN')
parser.add_argument('--dnn_dropout_rate', type=float, default=0.5, help='Dropout rate for DNN')


# CNN
parser.add_argument('--cnn_filters', type=int, defulat=16, help='Number of filters for CNN')
parser.add_argument('--cnn_kernel_size', type=int, default=3, help='Number of kernel size for CNN')
parser.add_argument('--cnn_pool_size', type=int, default=2, help='Number of pool size for CNN')
parser.add_argument('--cnn_strides', type=int, default=2, help='Number of strides for CNN')
parser.add_argument('--cnn_dense1', type=int, default=512, help='Number of 1st dense unit for CNN')
parser.add_argument('--cnn_dense2', type=int, default=128, help='Number of 2nd dense unit for CNN')
parser.add_argument('--cnn_dense3', type=int, default=2, help='Number of 3rd dense unit for CNN')
parser.add_argument('--cnn_dropout_rate', type=float, default=0.5, help='Dropout rate for CNN')


# ResNet
parser.add_argument('--resnet_filters', type=int, default=16, help='Number of filters for ResNet')
parser.add_argument('--resnet_kernel_size', type=int, default=3, help='Number of kernel size for ResNet')
parser.add_argument('--resnet_dense1', type=int, default=512, help='Number of 1st dense unit for ResNet')
parser.add_argument('--resnet_dense2', type=int, default=128, help='Number of 2nd dense unit for ResNet')
parser.add_argument('--resnet_dense3', type=int, default=2, help='Number of 3rd dense unit for ResNet')
parser.add_argument('--resnet_dropout_rate', type=float, default=0.5, help='Dropout rate for ResNet')
