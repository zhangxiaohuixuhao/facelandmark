import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image




def load_data(phase):
    data_file = phase + '.txt'
    with open(data_file) as f:
        line = f.readlines() # 查看返回值
    if phase == 'Train' or phase == 'train':
        transform = transforms.Compose([
            Normalize(),                # do channel normalize
            ToTensor()                  # convert to torch type: N*C*H*W
            ])
    else:
        transform = transforms.Compose([
            Normalize(),                # do channel normalize
            ToTensor()                  # convert to torch type: N*C*H*W
            ])
    dataset = FaceLandmarkDataset(line, phase, transform)
    return dataset


def get_train_test_data():
    train_set = load_data('train')
    test_set = load_data('test')
    return train_set, test_set

def main():
    # TODO
    # show landmark
    print('ok')

if __name__ == "__main__":
    main()