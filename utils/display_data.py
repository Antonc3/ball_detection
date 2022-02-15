import cv2
import glob
import os
import time

#save labeled image into a folder
save_image = True
image_out_path = 'data\labeled_images'
img_path = 'img'
ann_path = 'ann/yolo'
dirs = ['data/1477_balls','data/my_dataset']
classes = ['cargo_red', 'cargo_blue']
image_type = "png"
clr = [[255,0,0], [0,0,255]]
print(clr)
def getImagesInDir(path):
    imgs = []
    for filename in glob.glob(path+'/*.'+image_type):
        imgs.append(filename)
    return imgs

def displayImage(dir,img_name):
    basename = os.path.basename(img_name) 
    img = cv2.imread(img_name)
    annotation = os.path.join(dir+"/"+ann_path,os.path.splitext(basename)[0]+'.png.txt')
    h,w,c = img.shape
    label = open(annotation,'r').read().splitlines()
    for line in (label):
        data = line.split(' ')

        for d in range(len(data)):
            data[d] = float(data[d])
        pt1 = (int(w*(data[1]-data[3]/2)),int(h*(data[2]-data[4]/2)))
        pt2 = (int(w*(data[1]+data[3]/2)),int(h*(data[2]+data[4]/2)))
        # print(f"data: {data[0]}, type(data){type(data[0])}")
        # print(f"int data: {int(data[0])}, type(int(data)){type(int(data[0]))}")
        # print(f"clr[int(data[0])], {clr[int(data[0])]}, type(clr[int(data[0])]) {type(clr[int(data[0])])}")
        # print(clr[int(data[0])])
        cv2.rectangle(img,pt1,pt2,color=clr[(int(data[0])+1)%2],thickness=1)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    if save_image:
        cv2.imwrite(os.path.join(image_out_path,basename),img)
    return img

cwd = os.getcwd()

for dir in dirs:
    print(cwd+'/'+dir+'/'+img_path)
    imgs = getImagesInDir(cwd+'/'+dir+'/'+img_path)
    for img in imgs:
        displayImage(dir,img)
    cv2.destroyAllWindows()
    print("finished processing directory ",dir)
