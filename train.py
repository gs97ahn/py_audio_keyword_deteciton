import os
import shutil

from config.config import get_config
from utils.cnn import CNN
from utils.dnn import DNN
from utils.resnet import ResNet

config = get_config()

if __name__ == '__main__':
    # cnn or dnn or resnet
    model_type = config.model
    model = None
    if model_type == 'cnn':
        model = CNN()
        print('*** CNN Training Start ***')
    elif model_type == 'dnn':
        model = DNN()
        print('*** DNN Training Start ***')
    elif model_type == 'resnet':
        model = ResNet()
        print('*** ResNet Training Start ***')
    else:
        print('Error: Incorrect model inputted')
        exit(1)

    # remove model folder if exist
    if os.path.isdir(config.model_folder_path):
        shutil.rmtree(config.dataset)
    # create a new model folder
    os.mkdir(config.dataset)

    model.train()
    print('*** Training End ***')
