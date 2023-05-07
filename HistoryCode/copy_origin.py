import cv2
import xml.etree.ElementTree as ET
import os,sys
from albumentations import  HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, \
    RandomRotate90, Transpose, ShiftScaleRotate, Blur, CenterCrop, RandomCrop, \
    OpticalDistortion, GridDistortion, HueSaturationValue, \
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, \
    IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, GaussianBlur,\
    RandomBrightness, Flip, OneOf, VerticalFlip, Resize, Rotate, Compose,RandomBrightnessContrast,RGBShift,Cutout,CoarseDropout

import numpy as np
from tqdm import tqdm

import shutil
Flag_copy = 1           # 0=不复制 1=复制 复制原始图片===>图像增强文件夹 
Flag_run_or_test = 1

path_myadd = '_X1_1100'  # 注意后缀名   '_Rb_Sort' 
jpgPath = 'JPEGImages'  +  path_myadd  
xmlPath = 'Annotations' +  path_myadd  
savepath_root = 'IMG_AUG' 

AUGstr_list = os.listdir('IMG_AUG')

for i_str in range(len(AUGstr_list)):
    if Flag_copy:   # 复制原始图片===>图像增强文件夹   1=复制 2=不复制
        # 复制原始图像 ==> 图像增强
        SourcePath_IMGpath = jpgPath
        SourcePath_xmlPath = xmlPath
        Folder = 'Origin'
        if Flag_run_or_test:
            Target_anno_dir = os.path.join(savepath_root,  AUGstr_list[i_str], 'Annotations_' + 'AUG' ) 
            Target_img_dir = os.path.join(savepath_root, AUGstr_list[i_str], 'JPEGImages_' + 'AUG' ) 
        else: 
            Target_anno_dir = os.path.join(savepath_root, Folder, 'Annotations_' + 'AUG' ) 
            Target_img_dir =  os.path.join(savepath_root, Folder, 'JPEGImages_' + 'AUG' ) 

        

        shutil.copytree(SourcePath_xmlPath, Target_anno_dir, dirs_exist_ok=True)
        shutil.copytree(SourcePath_IMGpath, Target_img_dir , dirs_exist_ok=True)
        print('copytree == OK')
        print('NUM_IMG_After_Data_Augmentation', len(os.listdir(SourcePath_IMGpath)) * 8)