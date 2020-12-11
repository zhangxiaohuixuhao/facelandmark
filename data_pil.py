import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


train_boarder = 112

def channel_norm(img):
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.000000001)
    return pixels


class FaceLandmarksDataset(Dataset):
    def __init__(self, src_lines, transform=None):
        self.lines = src_lines
        self.transform = transform
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        img_name, rext, landmarks = parse_line(self.line[idx])
        






class Normalize(object):
    '''
    resize Image to train_boarder * train_boarder. Here we use 112 * 112
    then do channel normalizations:(image - mean)/sta_variation
    '''
    def __call__(self, sample):
        image,landmarks = sample['image'], sample['landmarks']
        image_resize = np.asarray(image.resize((train_boarder, train_boarder), Image.BILINEAR),
                                dtype=np.float32) #Image的resize并转换成array
        image = channel_norm(image_resize)
        return {'image': image, 'landmarks': landmarks}
        

class ToTensor(object):
    '''
    Convert arrays in sample to Tensor.
    Tensor channel sequence : N * C * H * W
    '''
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # numpy image: h*w*c
        # torch image:c*h*w

        # array的转置
        image = image.transpose((2,0,1)) 
        # torch : batch_size * c * h * w
        # 因此需要在array前面加一个通道
        image = np.expand_dims(image, axis=0) 
        return {'image':torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


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