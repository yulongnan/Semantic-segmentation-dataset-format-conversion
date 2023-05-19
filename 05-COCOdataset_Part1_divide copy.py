import os
import glob
import shutil
from tqdm import tqdm
# COCO数据集文件夹路径创建  ADE20 与 COCO 
data_root = 'IMG_AUG_COCO/COCOData_HLGSeg'    
img_dir_train = 'images/train2017'   
ann_dir_train ='labels/train2017_mask_graymmseg'   

img_dir_val='images/val2017'  
ann_dir_val='labels/val2017_mask_graymmseg'  

for i_path in [img_dir_train, ann_dir_train, img_dir_val, ann_dir_val]:
    i_path_all = os.path.join(data_root, i_path) 
    if not os.path.exists(i_path_all): 
        os.makedirs(i_path_all)

# 原始=主路径 
train_add_aug = True #含增强 True / 不含增强 False
if train_add_aug:
    Ori_jpgPath = 'IMG_AUG\JPEGImages_AUG_image'  
    Ori_maskpngPath = 'IMG_AUG\SegmentationClass_AUG_mask_graymmseg' 
else:
    Ori_jpgPath = 'JPEGImages'  +  '_X1_1370'   
    Ori_maskpngPath = 'SegmentationClass' + '_pseudo_1370' 
Imgname_prefix = 'RbRGB_pitaya_' 


# 训练集 --COCO Dataset-- train + AUG ==> augtrain.txt  
# 获得训练图像的编号  
with open('ImageSets\Segmentation\\augtrain.txt','r') as f: 
    train_lines = f.readlines() 

imgtrain_name_list = [] 
for i_num in range(len(train_lines)): 
    imgtrain_name_list.append(train_lines[i_num][:-1]) 
# print(len(imgtrain_name_list)) # 测试
# print(imgtrain_name_list[:5])  # 测试

# 复制 == 图像 + mask
for imgname in tqdm(imgtrain_name_list): 
    # 原始路径 = 图片+mask 
    src_img = os.path.join(Ori_jpgPath, imgname + '.jpg') 
    src_mask = os.path.join(Ori_maskpngPath, imgname + '.png')  

    # 保存路径 = 图片+mask 
    dst_img = os.path.join(data_root, img_dir_train, imgname + '.jpg') 
    dst_mask = os.path.join(data_root, ann_dir_train, imgname + '.png') 

    # 复制操作 == 图片 + mask 
    shutil.copy(src_img, dst_img) 
    shutil.copy(src_mask, dst_mask) 


# # 测试集 -- --COCO Dataset-- val  ==> originval.txt 
# 获得测试图像的编号 
with open('ImageSets\Segmentation\originval.txt','r') as f:
    val_lines = f.readlines() 
imgval_name_list = []
for i_num in range(len(val_lines)):
    imgval_name_list.append(val_lines[i_num][:-1])
# print(len(imgval_name_list)) # 测试
# print(imgval_name_list[:5])  # 测试
# 复制 == 图像 + mask 
for num_img in tqdm(imgval_name_list): 
    # 原始路径 = 图片 + mask 
    src_img = os.path.join(Ori_jpgPath, num_img + '.jpg') 
    src_mask = os.path.join(Ori_maskpngPath, num_img + '.png')  

    # 保存路径 = 图片 + mask 
    dst_img = os.path.join(data_root, img_dir_val, num_img + '.jpg') 
    dst_mask = os.path.join(data_root, ann_dir_val, num_img + '.png') 

    # 复制操作 == 图片 + mask  
    shutil.copy(src_img, dst_img) 
    shutil.copy(src_mask, dst_mask) 


