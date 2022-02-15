import cv2
import keyboard
import glob
import os
import numpy as np
annotation_path = 'data\labels'
image_path = 'data\images'
colors = [np.array(255,0,0), np.array(0,255,0)]
def getImagesInDir(path):
    imgs = []
    for filename in glob.glob(path+'/*.jpg'):
        imgs.append(filename)
    return imgs

def displayImage(img_name):
    basename = os.path.basename(img_name)
    img = cv2.imread(os.path.join(image_path,img_name))
    annotation = os.path.join(annotation_path,basename+'.txt')
    w,h,c = img.shape
    label = open(annotation,'r').read().splitlines()
    for line in range(len(label)):
        data = line.split(' ')

        for d in range(len(data)):
            data[d] = float(data[d])
        pt1 = (w*(data[1]-data[3]/2),h*(data[2]-data[4]/2))
        pt2 = (w*(data[1]+data[3]/2),h*(data[2]+data[4]/2))(w)
        cv2.rectangle(img,pt1,pt2,colors[data[0]],thickness=1)
    cv2.imshow(basename, img)


imgs = getImagesInDir(image_path)
cur = 0
while True: 
    cur_img = imgs[cur]
    if keyboard.is_pressed('space'):
        cur = cur+1
    displayImage(cur_img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()