'''
@lanhuage: python
@Descripttion: Deprecated. Just for test. For more information,see convertmask.utils.mask2json_script.
@version: beta
@Author: xiaoshuyui
@Date: 2020-06-09 16:24:12
LastEditors: xiaoshuyui
LastEditTime: 2020-11-09 13:58:09
'''
import sys

import cv2

sys.path.append('..')
import glob 
import json 
import os 
from tqdm import tqdm

from convertmask.utils.methods import get_multi_shapes

def getJsons(imgPath, maskPath, savePath, yamlPath=''):
    """
    imgPath: origin image path \n
    maskPath : mask image path \n
    savePath : json file save path \n
    
    >>> getJsons(path-to-your-imgs,path-to-your-maskimgs,path-to-your-jsonfiles) 

    """
    oriImgs = glob.glob(imgPath + os.sep + '*.jpg')

    for i_img in tqdm(oriImgs):
        
        # print(i_img)
        # print(os.path.basename(i_img).split('.')[0]+'.png') 
        name_mask = os.path.basename(i_img).split('.')[0]+'.png'  # 获得img对应的mask名字
        i_mask = os.path.join(maskPath, name_mask)                # mask的路径
        # print(i_mask)

        get_multi_shapes.getMultiShapes(i_img, i_mask, savePath, yamlPath) # 对每张图片


if __name__ == "__main__":

    root_path = 'IMG_AUG_COCO\COCOData_HLGSeg' 
    
    # 1 == 训练集 train greymask2json 
    # # 路径 = train     annotations
    imgPath = os.path.join(root_path,'images\\train2017' ) 
    maskPath= os.path.join(root_path,'labels\\train2017_mask_graymmseg' ) 
    savePath_train= os.path.join(root_path,'labelme_JSON\\train2017_json_labelme' ) 
    yamlPath='info_HLGseg.yaml' 
    
    # 生成保存文件夹
    os.makedirs(savePath_train, exist_ok=True) 
     
    # 对文件夹中所有mask 执行 masktojson   #mask是灰度图 
    getJsons(imgPath, maskPath, savePath_train, yamlPath) 


    # 2 == 训练集 val greymask2json 
    # # 路径 = val     annotations 
    imgPath = os.path.join(root_path,'images\\val2017' ) 
    maskPath= os.path.join(root_path,'labels\\val2017_mask_graymmseg' ) 
    savePath_val= os.path.join(root_path,'labelme_JSON\\val2017_json_labelme' ) 
    yamlPath='info_HLGseg.yaml' 
    
    # 生成保存文件夹
    os.makedirs(savePath_val, exist_ok=True) 
     
    # 对文件夹中所有mask 执行 masktojson   #mask是灰度图 
    getJsons(imgPath, maskPath, savePath_val, yamlPath) 