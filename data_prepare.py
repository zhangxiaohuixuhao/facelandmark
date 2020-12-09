import torch
import numpy as np
import os
import cv2
import shutil
import glob


img_path = 'E:/code/case1/data/I/'
lab_path = 'E:/code/case1/facelandmark/landmark.txt'
save_path = 'E:/class_hexin/projectII_face_keypoints_detection/data.zip/data/result/'
data_path = 'E:/code/case1/data/data_end/'
end_lab = 'E:/code/case1/facelandmark/label.txt'
def data_prepare(data_path, lab_path, end_lab):
    f = open(lab_path, 'r')
    img_names = os.listdir(img_path)
    lab_mark = open(lab_path).read().strip().split('\n')
    landmark_f = open(end_lab, 'w+')
    for landmark in lab_mark:
        lab_infor = landmark.split()
        lab_infor[0] = data_path + lab_infor[0]
        for temp in lab_infor:
            landmark_f.write(temp + ' ')
        landmark_f.write('\n')

        

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
            if temp != None:
                cv2.imwrite(save_path + temp, img)
                print('ok')
            img = cv2.imread(img_path + img_name)
        face_x1, face_y1, face_x2, face_y2 = int(float(lab_infor[1])), int(float(lab_infor[2])), int(float(lab_infor[3])), int(float(lab_infor[4]))
        cv2.rectangle(img, (face_x1, face_y1), (face_x2, face_y2), (0, 255, 0), 1)
        for i in range(5, 47, 2):
            # cv2.circle(img, (int(float(lab_infor[i])), int(float(lab_infor[i + 1]))), 2, (0,0,255), -1)
            cv2.circle(img, (int(float(lab_infor[i]) + float(lab_infor[1])), int(float(lab_infor[i + 1]) + float(lab_infor[2]))), 2, (0,0,255), -1)
        temp = img_name

def expend_facedet(face_x1, face_y1, face_x2, face_y2, img):
    h, w = img.shape[:2]
    box_w = face_x2 - face_x1
    box_h = face_y2 - face_y1
    face_x1 = max(0, int(face_x1 - 0.25 * box_w))
    face_y1 = max(0, int(face_y1 - 0.25 * box_h))
    face_x2 = min(w, int(face_x2 + 0.25 * box_w))
    face_y2 = min(h, int(face_y2 + 0.25 * box_h))
    return face_x1, face_y1, face_x2, face_y2

def check_perlandmark(x, y, face_x1, face_y1, face_x2, face_y2, img):
    h, w = img.shape[:2]
    x = max(0, x)
    y = max(0, y)
    x = min(x, w)
    y = min(y, h)
    if x >= face_x1 and x <= face_x2 and y >= face_y1 and y <= face_y2:
        x = x - face_x1
        y = y - face_y1
        return True, x, y
    else:
        return False, x, y

def check_landmark1(img_path, lab_path, save_path):
    # 进行图像坐标的转换  人脸框 人脸框对应的人脸关键点坐标
    # 转换时查看人脸关键点坐标是否在人脸框中，若不在则不进行标签的写入
    # 进行转换后的坐标的保存
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    lab_mark = open(lab_path).read().strip().split('\n')
    temp = None
    img = None
    img_delet = []
    per_mark = []
    f = open('data2_endlandmark.txt', 'w+')
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
            mark_in_box, end_x, end_y = check_perlandmark(x, y, face_x1, face_y1, face_x2, face_y2, img)
            if mark_in_box:
                per_mark.append(str(max(end_x, 0)))
                per_mark.append(str(max(end_y, 0)))
            else:
                per_mark = []
                img_delet.append(img_name)
                break
        if len(per_mark):
            for per in per_mark:
                f.write(per + ' ')
            f.write('\n')
        per_mark = []
        temp = img_name
    f.close()
    print(img_delet)

def check_landmark(img_path, lab_path):
    # 查看标签与给定图像是否对应，若不对应则进行标签的删除
    f = open('data2.txt', 'w+')
    img_names = os.listdir(img_path)
    lab_mark = open(lab_path).read().strip().split('\n')
    for landmark in lab_mark:
        lab_infor = landmark.split()
        img_name = lab_infor[0]
        if img_name in img_names:
            f.write(landmark + '\n')

def copy_img_by_landmark(img_path, lab_path, data_path):
    lab_mark = open(lab_path).read().strip().split('\n')
    for landmark in lab_mark:
        lab_infor = landmark.split()
        img_name = lab_infor[0]
        if not os.path.exists(data_path + img_name):
            shutil.copy(img_path + img_name, data_path + img_name)

def main():
    # check_landmark(img_path, lab_path)
    # check_landmark1(img_path, 'data2.txt', save_path)
    # show_landmark(img_path, 'data2_endlandmark.txt', save_path)
    # copy_img_by_landmark(img_path, 'data1_endlandmark.txt', data_path)
    data_prepare(data_path, lab_path, end_lab)

if __name__=='__main__':
    main()