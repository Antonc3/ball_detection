import glob
import os
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join
import re
img_path = 'img'
ann_path = 'ann'
dirs = ['data/1477_balls','data/my_dataset']
classes = ['cargo_red', 'cargo_blue']


def getAnnotationInDir(dir_path):
    annotation_list = []
    for filename in glob.glob(dir_path + '/*.json'):
        annotation_list.append(filename)
    return annotation_list
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[2])/2.0 - 1
    y = (box[1] + box[3])/2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(dir_path, output_path, image_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]
    # print(image_path,basenames)
    in_file = open(image_path)
    lines = in_file.read().splitlines()
    size = [0,0]
    boxes = []
    for i, line in enumerate(lines):
        if line.find("width") >= 0:
            size[0] = int(re.findall("\d+",line)[0])
            continue
        if line.find("height") >= 0:
            size[1] = int(re.findall("\d+",line)[0])
            continue
        if line.find("exterior") >= 0:
            box = [0,0,0,0]
            box[0] = int(re.findall("\d+",lines[i+2])[0])
            box[1] = int(re.findall("\d+",lines[i+3])[0])
            box[2] = int(re.findall("\d+",lines[i+6])[0])
            box[3] = int(re.findall("\d+",lines[i+7])[0])
            boxes.append(box)
    out_file = open(output_path + "/"+basename_no_ext + '.txt', 'w')
    for box in boxes:
        bbox = convert(size,box)
        fin_box = []
        for i in range(4):
            fin_box.append(str(bbox[i]))
        out_file.write("0 " + ' '.join(fin_box) + '\n')
cwd = getcwd()
# full_dir_path = os.path.join(cwd, dirs[0])
# full_ann_path = os.path.join(full_dir_path, ann_path)
# output_path = full_ann_path + '/yolo'

# test_image = os.path.join(full_ann_path,"1477_balls_00000.png.json")
# convert_annotation(full_ann_path,output_path,test_image)


for dir_path in dirs:
    full_dir_path = os.path.join(cwd, dir_path)
    full_ann_path = os.path.join(full_dir_path, ann_path)
    output_path = full_ann_path + '/yolo'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    annotations = getAnnotationInDir(full_ann_path)
    for annotation in annotations:
        convert_annotation(full_ann_path,output_path,annotation)

    print("Finished processing: " + dir_path)
    