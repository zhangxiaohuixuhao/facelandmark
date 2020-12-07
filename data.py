import torch
import numpy as np
import os
import cv2
import shutil


img_path = 'E:/code/case1/data/I/'
lab_path = 'E:/code/case1/data/data1_label.txt'
save_path = 'E:/code/case1/data/result/'

def data_prepare(img_path, lab_path, train_lab, test_lab):
    print('ok')

# def show_landmark(img_path, lab_path, save_path):
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     lab_mark = open(lab_path).read().strip().split('\n')
#     temp = None
#     img = None
#     for landmark in lab_mark:
#         lab_infor = landmark.split()
#         img_name = lab_infor[0]
#         if temp != img_name:
#             if temp != None:
#                 cv2.imwrite(save_path + temp, img)
#                 print('ok')
#             img = cv2.imread(img_path + img_name)
#         # face_x1, face_x1, face_x1, face_x1 = float(lab_infor[1]), float(lab_infor[2]),
#         #                                     float(lab_infor[3]), float(lab_infor[4])
#         for i in range(5, 47, 2):
#             print(i)
#             cv2.circle(img, (int(float(lab_infor[i])), int(float(lab_infor[i + 1]))), 2, (0,0,255), -1)
#         temp = img_name
#         print('ok')

def expend_facedet(face_x1, face_y1, face_x2, face_y2, img):
    h, w = img.shape[:2]
    box_w = face_x2 - face_x1
    box_h = face_y2 - face_y1
    face_x1 = max(0, int(face_x1 - 0.25 * box_w))
    face_y1 = max(0, int(face_y1 - 0.25 * box_h))
    face_x2 = min(w, int(face_x2 + 0.25 * box_w))
    face_y2 = min(h, int(face_y2 + 0.25 * box_h))
    return face_x1, face_y1, face_x2, face_y2


def check_perlandmark(x, y, face_x1, face_y1, face_x2, face_y2):
    if x >= face_x1 and x <= face_x2 and y >= face_y1 and y <= face_y2:
        x = x - face_x1
        y = y - face_y1
        return True, x, y
    else:
        return False, x, y

def check_landmark1(img_path, lab_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    lab_mark = open(lab_path).read().strip().split('\n')
    temp = None
    img = None
    per_mark = []
    f = open('data1_endlandmark.txt', 'w+')
    for landmark in lab_mark:
        print()
        lab_infor = landmark.split()
        img_name = lab_infor[0]
        per_mark.append(img_name)
        if temp != img_name:
            img = cv2.imread(img_path + img_name)
        face_x1, face_y1, face_x2, face_y2 = expend_facedet(float(lab_infor[1]), float(lab_infor[2]),
                                            float(lab_infor[3]), float(lab_infor[4]), img)
        per_mark.append(str(face_x1))
        per_mark.append(str(face_y1))
        per_mark.append(str(face_x2))
        per_mark.append(str(face_y2))
        for i in range(5, 47, 2):
            x, y = int(float(lab_infor[i])), int(float(lab_infor[i + 1]))
            mark_in_box, end_x, end_y = check_perlandmark(x, y, face_x1, face_y1, face_x2, face_y2)
            if mark_in_box:
                per_mark.append(str(end_x))
                per_mark.append(str(end_y))
            else:
                per_mark = []
                print(img_name)
                break
        if len(per_mark):
            for per in per_mark:
                f.write(per + ' ')
            f.write('\n')
        per_mark = []
        temp = img_name
    f.close()

def check_landmark(img_path, lab_path):
    f = open('data1.txt', 'w+')
    img_names = os.listdir(img_path)
    lab_mark = open(lab_path).read().strip().split('\n')
    for landmark in lab_mark:
        lab_infor = landmark.split()
        img_name = lab_infor[0]
        if img_name in img_names:
            f.write(landmark + '\n')
    
def main():
    # check_landmark(img_path, lab_path)
    # check_landmark1(img_path, 'data1.txt', save_path)
    show_landmark(img_path, 'data1.txt', save_path)
    # data_prepare(img_path, lab_path, train_lab, test_lab)
    # show_datapre(img_path, train_lab, test_lab)

if __name__=='__main__':
    main()