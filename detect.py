import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import tensorrt as trt
import time
import numpy as np
import ctypes
import socket


CONF_THRESH=0.5
IOU_THRESHOLD=0.4

MIN_HBLUE = 80
MAX_HBLUE = 120

MIN_HRED1 = 0
MAX_HRED1 = 20
MIN_HRED2 = 165
MAX_HRED2 = 180

BALL_F = 653.8
BALL_SIZE = 9.5

WH_RATIO = 2

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

engine_path = "model/exp12.engine"

RED = 0
BLUE = 1

COLOR_THRESH = 0

colors = [(255,0,0),(0,0,255)]

class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('binding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)


        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size
    
    def infer(self, image):
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        context = self.context
        stream = self.stream

        img, h, w = self.preprocess_image(image)
        np.copyto(host_inputs[0],img.ravel())
        start = time.time()
        cuda.memcpy_htod_async(cuda_inputs[0],host_inputs[0],stream=stream)
        context.execute_async(batch_size=1,bindings=bindings,stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0],cuda_outputs[0],stream=stream)
        stream.synchronize()
        end = time.time()
        print("time taken {}".format(end-start))
        output = host_outputs[0]
        
        results_boxes,results_scores,result_classid = self.post_process(output[0:6001],h,w)
        return results_boxes, results_scores, result_classid

    def preprocess_image(self, raw_image):
        
        h,w,c = raw_image.shape
        image = cv2.cvtColor(raw_image,cv2.COLOR_BGR2RGB)
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
        )

        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image,[2,0,1])
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image, h, w

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
   
    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # Do nms
        boxes = self.non_max_suppression(pred, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))            
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > CONF_THRESH
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Transform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes

def calc_ball_dist(width):
    return BALL_F*BALL_SIZE/width

def calc_degree(dist, dist_to_center):
    return np.floor(np.arcsin(dist_to_center/dist)*180/3.14)

def calc_ball_to_center(bbox, screen_width):
    ball_dist_to_center_pix = np.abs(screen_width/2-(bbox[2]-bbox[0])/2)
    ball_dist_to_center_inc = BALL_SIZE*ball_dist_to_center_pix/(bbox[2]-bbox[0])
    return ball_dist_to_center_inc

def findColor(img,bbox):
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    hmean = np.mean(hsv[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),0])
    if (hmean > MIN_HRED1 and hmean < MAX_HRED1) or (hmean > MIN_HRED2 and hmean < MAX_HRED2):
        return RED
    if (hmean > MIN_HBLUE and hmean < MAX_HBLUE):
        return BLUE
    return -1

def draw_box(img, box, color,label):
    imgHeight, imgWidth, _ = img.shape
    thick = (imgHeight + imgWidth) // 900
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thick)
    cv2.putText(img, label, (int(box[0]), int(max(0,box[1] - 12))), 0, 1e-3 * imgHeight, color, 2)

def convert_string(balls):
    ballstr = str(len(balls))
    for ball in balls:
        ballstr += " " + str(ball[0]) + " " + str(ball[1])
    return ballstr

HOST = "10.1.14.2"
PORT = 8888
HOST = "10.220.8.28"
PLUGIN_LIBRARY = "build/libmyplugins.so"

ctypes.CDLL(PLUGIN_LIBRARY)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAMERA_HEIGHT)
inf = YoLov5TRT(engine_path)

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST,PORT))
s.listen()
conn, addr = s.accept()
while True:
    timer = cv2.getTickCount()
    ret, frame = cap.read()
    bboxes, conf, classes  = inf.infer(frame)
    # # print(bboxes.ravel())
    finalbboxes = []
    # format of distance, degree
    finalballs = []
    for i, bbox in enumerate(bboxes):
        clr = findColor(frame,bbox)
        if clr == -1:
            continue
        width = bbox[3]-bbox[1]
        height = bbox[2]-bbox[0]
        if width*WH_RATIO < height or height*WH_RATIO<width:
            continue
        dist = calc_ball_dist(max(width,height))
        deg = calc_degree(dist,calc_ball_to_center(bbox,CAMERA_WIDTH))
        label = "dist: " + str(np.floor(dist)) + " deg: " + str(np.floor(deg))
        finalbboxes.append(bbox)
        finalballs.append((np.floor(dist),np.floor(deg)))        
        draw_box(frame,bbox,colors[clr],label)
    
    conn.send(bytes(convert_string(finalballs),'utf-8'))
    fps = cv2.getTickFrequency() / (cv2.getTickCount()-timer)
    # cv2.putText(frame,str(int(fps)), (0,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.75,(0,255,0),2)
    print(f"fps: {fps}")
    cv2.imshow("frame", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
inf.destroy()
conn.close()

