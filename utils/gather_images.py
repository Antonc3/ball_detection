import uuid
import cv2
import os
import keyboard
import time

IMAGES_PATH = os.path.join('data', 'images2')
cap = cv2.VideoCapture(0)
# Loop through labels
img_number = 0
img_type = 'nothing'
while cap.isOpened():
    
    ret,frame = cap.read()
    cv2.imshow('Image Collection', frame)

    if keyboard.is_pressed('space'): 
        imgname = os.path.join(IMAGES_PATH, img_type+'_'+str(img_number)+'.jpg')
        cv2.imwrite(imgname,frame)
        img_number = img_number+1
        print('Image Number: {}; Name: {}'.format(img_number,imgname))
        time.sleep(.5)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()