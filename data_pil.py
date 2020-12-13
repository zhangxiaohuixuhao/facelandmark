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

def parse_line(line):
    line_parts = line.strip().split()
    img_name = line_parts[0]
    rect = list(map(int, list(map(float, line_parts[1:5]))))
    landmarks = list(map(float, line_parts[5: len(line_parts)]))
    return img_name, rect, landmarks

class FaceLandmarkDataset(Dataset):
    def __init__(self, src_lines, transform=None):
        self.lines = src_lines
        self.transform = transform
    
    def __len__(self):
        return len(self.lines)
    
    def getitem(self):
        img_name, rect, landmarks = parse_line(self.lines[0]) # 问题这函数在处理的时候是针对于一个样本处理的，为什么不需要进行循环呢？
        image = Image.open(img_name).convert('L') # convert函数用来转换图像的格式，其中‘L’表示转成灰度图像
        img_crop= image.crop(tuple(rect))
        landmarks = np.array(landmarks).astype(np.float32)
        for i in range(0, 42, 2):
            landmarks[i] = train_boarder * landmarks[i] / (rect[2] - rect[0])
            landmarks[i + 1] = train_boarder * landmarks[i + 1] / (rect[3] - rect[1])
        landmarks = np.expand_dims(landmarks, axis=0)
        temp_sample = {'image': img_crop, 'landmarks': landmarks}
        temp_sample = self.transform(temp_sample)    
        for idx in range(1, len(self.lines)):
            img_name, rect, landmarks = parse_line(self.lines[idx]) # 问题这函数在处理的时候是针对于一个样本处理的，为什么不需要进行循环呢？
            image = Image.open(img_name).convert('L') # convert函数用来转换图像的格式，其中‘L’表示转成灰度图像
            img_crop= image.crop(tuple(rect))
            landmarks = np.array(landmarks).astype(np.float32)
            for i in range(0, 42, 2):
                landmarks[i] = train_boarder * landmarks[i] / (rect[2] - rect[0])
                landmarks[i + 1] = train_boarder * landmarks[i + 1] / (rect[3] - rect[1])
            landmarks = np.expand_dims(landmarks, axis=0)
            sample = {'image': img_crop, 'landmarks': landmarks}
            sample = self.transform(sample)
            temp_sample['image'] = torch.cat((temp_sample['image'], sample['image']), dim=0)
            temp_sample['landmarks'] = torch.cat((temp_sample['landmarks'], sample['landmarks']), dim=0)
        return temp_sample

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
        # image = image.transpose((2,0,1)) 
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
    dataset = FaceLandmarkDataset(line, transform)
    sample = dataset.getitem()
    return sample


def get_train_test_set():
    train_set = load_data('train_pc')
    test_set = load_data('test_pc')
    return train_set, test_set


def main():
    # TODO
    # show landmark
    # lines = open('train.txt').read().strip().split('\n')
    # for line in lines:
    # img_name, rect, landmarks = parse_line(line)
    train_set, test_set = get_train_test_set()
    print('ok')

if __name__ == "__main__":
    main()