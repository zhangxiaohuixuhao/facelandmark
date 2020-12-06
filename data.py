import torch
import numpy as np
import os
import cv2


img_path = 'E:/class_hexin/projectII_face_keypoints_detection/data.zip/data/I/'
lab_path = 'E:/class_hexin/projectII_face_keypoints_detection/data.zip/data/I/label.txt'
save_path = './result/'

def data_prepare(img_path, lab_path, train_lab, test_lab):
    print('ok')

def show_landmark(img_path, lab_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    lab_mark = open(lab_path).read().strip().split('\n')
    temp = None
    img = None
    for landmark in lab_mark:
        lab_infor = landmark.split()
        img_name = lab_infor[0]
        if temp != img_name:
            img = cv2.imread(img_path + img_name)
        face_x1, face_x1, face_x1, face_x1 = float(lab_infor[1]), float(lab_infor[2]),
                                            float(lab_infor[3]), float(lab_infor[4])
        for i in range(5, 26):
            cv2.circle(img, (int(lab_infor[i+ 4]), int(lab_infor[i + 4])), 2, (0,0,255), -1)
        temp = img_name
        print(lab_infor)
    print('ok')


def main():
    show_landmark(img_path, lab_path, save_path)
    data_prepare(img_path, lab_path, train_lab, test_lab)
    show_datapre(img_path, train_lab, test_lab)

if __name__=='__main__':
    main()