import glob
import os
import xml.etree.ElementTree as ET
import re
import numpy as np
img_path = 'new_imgs'
ann_path = 'new_labels'
dirs = ['data/']
# classes = ['cargo_red', 'cargo_blue']

def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.png'):
        image_list.append(filename)
    return image_list
def getAnnotationInDir(dir_path):
    annotation_list = []
    for filename in glob.glob(dir_path + '/*.json'):
        annotation_list.append(filename)
    return annotation_list
cwd = os.getcwd()

for dir_path in dirs:
    dir_name = dir_path[5:]
    print(dir_name)
    full_dir_path = os.path.join(cwd, dir_path)
    full_img_path = os.path.join(full_dir_path, img_path)
    full_ann_path = os.path.join(full_dir_path, ann_path)
    print(full_img_path)
    images = getImagesInDir(full_img_path)
    print(len(images))
    for image_path in images:
        print(image_path)
        basename = os.path.basename(image_path)

        #THIS IS WHERE YOU DETERMINE HOW YOU WANT TO CHANGE THE NAME
        new_base = os.path.splitext(os.path.splitext(basename)[0])[0]
        # print(basename+'.txt',new_base+'.txt')
        # print(os.path.join(full_ann_path,basename + '.txt'),os.path.join(full_ann_path,new_base+'.txt'))
        
        # file_num = re.split('frame', basename)[1]
        # new_file_name = dir_name + file_num
        # print(new_file_name)
        # os.rename(os.path.join(full_img_path,basename),os.path.join(full_img_path,new_file_name))
        os.rename(os.path.join(full_ann_path,basename + '.txt'),os.path.join(full_ann_path,new_base+'.txt'))

    print("Finished processing: " + dir_path)
    