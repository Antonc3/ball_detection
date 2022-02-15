import torch
import cv2
import numpy as np
import glob
import os

model = torch.hub.load('yolov5', 'custom', path='model/exp12weights.pt',source='local',force_reload=True)

input_directory = ['data/images']
output_directory = 'data/exp7'

def getImagesInDir(dir):
    imgs = []
    for img in glob.glob(dir+"/*.jpg"):
        imgs.append(img)
    for img in glob.glob(dir+ "/*.png"):
        imgs.append(img)
    return imgs


def detectImages(input_dir,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    imgs = getImagesInDir(input_dir)
    print("images length",len(imgs))
    for img in imgs:
        image = cv2.imread(img)
        result = model(image)
        basename = os.path.basename(img)
        cv2.imwrite(os.path.join(output_dir,basename),np.squeeze(result.render()))


def testCamera(cam):
    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret: 
            results = model(frame)
            print(results.pandas().xyxy)
            cv2.imshow('Camera', np.squeeze(results.render()))
            if cv2.waitKey(10) & 0xFF == ord('q'): 
                break   
    cap.release()
    cv2.destroyAllWindows()
def testVideo(vid):
    cap = cv2.VideoCapture(vid)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret: 
            results = model(frame)
            print(results.pandas().xyxy)
            cv2.imshow('Video', np.squeeze(results.render()))
            if cv2.waitKey(10) & 0xFF == ord('q'): 
                break
    cap.release()
    cv2.destroyAllWindows()

# for dir in input_directory:
#     detectImages(dir,output_directory)

testCamera(0)
# testVideo('FIRST game piece_ds0_2021-11-19 09-33-03.mp4')
    # Make detections 
    # results = model(frame)
    # print(results)
    # cv2.imshow('YOLO', np.squeeze(results.render()))
    
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break
