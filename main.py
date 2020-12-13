# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import cv2
from data_pil import get_train_test_set


# torch.set_default_tensor_type(torch.FloatStorage)
def train(train_loader, test_loader, model, criterion_pts, optimizer):
    print('ok')

def train(args, train_loader, test_loader, model, criterion, optimizer, device):
    if args.save_model:
        if os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)
    
    epochs = args.epochs

    train_losses = []
    valid_losses = []

    for epoch_id in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0

        # train model
        model.train() # 使用BatchNormalizetion()和Dropout()
        for batch_idx, batch in enumerate(train_loader):
            img = batch['image']
            landmark = batch['landmarks']
            
            # ground truth
            input_img = img.to(device)
            target_pts = landmark.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # get out_put
            output_pts = model(imput_img)

            loss = criterion(output_pts, target_pts)

            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t pts_loss: {:.6f}'.format(
                    epoch_id,
                    batch * len(img),
                    len(train_loader.dataset),
                    100 * batch_idx / len(train_loader),
                    loss.item()
                ))
        
        ######################
        # validate the model
        ######################
        valid_mean_pts_loss = 0.0

        model.eval() # 不使用BatchNormalization()和Dropout() pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。
        with torch.no_grad():
            valid_batch_cnt = 0

            for valid_batch_idx, batch in enumerate(test_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                landmark = batch['landmarks']

                input_img = valid_img.to(device)
                target_pts = landmark.to(device)

                output_pts = model(input_img)

                valid_loss = criterion(output_pts, target_pts)

                valid_mean_pts_loss += valid_loss.item()
            
            valid_mean_pts_loss /= valid_batch_cnt * 1.0
            print('Valid: pts_loss: {:.6f}'.format(valid_mean_pts_loss))
        
        print('==========================================================')

        if args.save_model:
            saved_model_name = os.path.join(args.save_directory, 'facelandmark_epoch' + '_' + str(epoch_id) + '.pth')
            torch.save(model.state_dict(), saved_model_name)

def main():
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default 64)')
    parser.add_argument('--test_batch_size', type=int, default=64, metavar='N', help='input batch size for testing(default 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train(default 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate(default 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=True, help='disables CUDAtraining')
    parser.add_argument('--log_interval', type=int, default=20, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=True, help='save the current model')
    parser.add_argument('--save_directory', type=str, default='trained_models', help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='train', help='training, predicting or finetuning')

    args = parser.parse_args()
    # torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torh.cuda.is_available()
    # for single GPU
    device = torch.device('cuda' if use_cuda else 'cpu')
    # For multi GPUs, nothing need to change here
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    print('===========> Loading Datasets')
    train_set, test_set = get_train_test_set()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)

    print('===========> Building Model')
    #TODO
    # For single model
    model = Net().to(device)
    ####################################################
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    ####################################################
    if args.phase == 'Train' or args.phase == 'train':
        print('========>Start Traning')
        train(args, train_loader, test_loader, model, criterion, optimizer, device)
    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Test')
        # how to do test?
    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Finetune')
        # how to do finetune?
    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Predict')
        # how to do predict?

if __name__=='__main__':
    main()