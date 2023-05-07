import os
import shutil

path_root = '../Data_Augmentation_VOC_HLGSeg/'
copy_path = [name for name in os.listdir(path_root)if os.path.isdir(os.path.join(path_root, name))]
for i_path in copy_path:
    os.makedirs(os.path.join('../Data_Augmentation_VOC_HLGSeg_Test1/', i_path) , exist_ok=True)


