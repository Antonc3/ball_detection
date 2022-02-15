import glob
import os

#save labeled image into a folder
save_image = True
annotation_path = 'data\labels'
annotation_out_path = 'data\labels2'
image_path = 'data\images'
image_out_path = 'data\labeled_images'
clr = [[255,0,0], [0,0,255]]
print(clr)
def getImagesInDir(path):
    imgs = []
    for filename in glob.glob(path+'/*.jpg'):
        imgs.append(os.path.basename(filename))
    return imgs

def write_annotation(img_name):
    basename = os.path.basename(img_name) 
    annotation = os.path.join(annotation_path,os.path.splitext(basename)[0]+'.txt')
    print(os.path.join(annotation_out_path,annotation))
    file = open(os.path.join(annotation_out_path,os.path.splitext(basename)[0]+'.txt'),'w')
    label = open(annotation,'r').read().splitlines()
    for line in (label):
        data = line.split(' ')
        data[0] = '0'
        line = ' '.join(data)
        print(line)
        line = line + '\n'  
        file.write(line)


imgs = getImagesInDir(image_path)
cur = 0
imgs.reverse()
cur_img = imgs[cur]
for img in imgs:
    write_annotation(img)