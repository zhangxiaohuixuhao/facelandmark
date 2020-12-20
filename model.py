# -*- coding: utf-8 -*-
import torch
import torchvision 
import torch.nn as nn

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=0)
        self.relu_conv1_1 = nn.PReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu_conv2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu_conv2_2 = nn.PReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3_1 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=0)
        self.relu_conv3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(in_channels=24,out_channels=24, kernel_size=3, stride=1, padding=0)
        self.relu_conv3_2 = nn.PReLU()
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv4_1 = nn.Conv2d(in_channels=24, out_channels=40, kernel_size=3, stride=1, padding=1)
        self.relu_conv4_1 = nn.PReLU()
        self.conv4_2 = nn.Conv2d(in_channels=40, out_channels=80, kernel_size=3, stride=1, padding=1)
        self.relu_conv4_2 = nn.PReLU()
        self.ip1 = nn.Linear(4*4*80, 128)
        self.relu_ip1  = nn.PReLU()
        self.ip2 = nn.Linear(128, 128)
        self.relu_ip2 = nn.PReLU()
        self.ip3 = nn.Linear(128, 42)
    
    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu_conv1_1(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.relu_conv2_1(x)
        x = self.conv2_2(x)
        x = self.relu_conv2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.relu_conv3_1(x)
        x = self.conv3_2(x)
        x = self.relu_conv3_2(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.relu_conv4_1(x)
        x = self.conv4_2(x)
        x = self.re.u_conv4_2(x)
        x = self.ip1(x)
        x = self.relu_ip1(x)
        x = self.ip2(x)
        x = self.relu_ip2(x)
        x = self.ip3(x)
        return x

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU()
        self.pooling1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU()
        self.pooling2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(1*1*120, 84)
        self.fc2 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pooling2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    



