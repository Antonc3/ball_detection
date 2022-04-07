import cv2

cap = cv2.VideoCapture(0)
print(cap.get(3),cap.get(4))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
while True:
    ret, frame = cap.read()
    print(ret)
    #h,w,c = frame.shape
    #print(h,w,c)
    #if ret:
     #   print(frame.shape())
    cv2.imshow("frame", frame)
    #print(frame.shape()[0], frame.shape()[1])
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

