from djitellopy import tello
import cv2
import numpy as np
import time
import logging
import torch
from ultralytics import YOLO
from supervision import Detections

tello.Tello.LOGGER.setLevel(logging.WARNING)

window_width, window_height = [640, 480]
fbRange = [6700, 8900]
pid = [0.4, 0.4, 0]
pError = 0

model = YOLO("model.pt")

def findFace(img):
    # 目标检测推理
    # frame = cv2.flip(img, 1)
    frame = img
    output = model(frame)
    results = Detections.from_ultralytics(output[0])

    # annotated_frame = output[0].plot()

    # 获取所有人脸矩形框的边界
    # x_min = float(1000)
    # y_min = float(1000)
    # x_max = float(-1000)
    # y_max = float(-1000)

    myFaceListC = []
    myFaceListArea = []

    for result in results.xyxy:

        # print(img_cvt)
        # 确保边界不超出原图像的尺寸
        x_min = int(max(result[0], 0))
        y_min = int(max(result[1], 0))
        x_max = int(min(result[2], window_width))
        y_max = int(min(result[3], window_height))

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        area = (x_max - x_min) * (y_max - y_min) 
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)

    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]

def trackFace(me, info, width, pid, pError):
    area = info[1]
    x, y = info[0]
    fb = 0

    error = x - width // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))

    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20

    if x == 0:
        speed = 0
        error = 0

    # print(speed, fb)

    me.send_rc_control(0, fb, 0, speed)
    return error

# cap = cv2.VideoCapture(1)
# cap.open(0)
# while cap.isOpened():

me = tello.Tello()
me.connect()
print("剩余电池电量：", me.get_battery())
me.streamon()

me.takeoff()
me.send_rc_control(0, 0, 40, 0)
time.sleep(3)
me.send_rc_control(0, 0, 0, 0)

last_time = time.time()

while True:
    # _, img = cap.read()
    cur_time = time.time()
    if cur_time - last_time > 1.0:
        print("当前电池电量：", me.get_battery())
        last_time = cur_time
    img = me.get_frame_read().frame
    # print("图像大小：", img.shape)
    img = cv2.resize(img, (window_width, window_height))
    img, info = findFace(img)
    pError = trackFace(me, info, window_width, pid, pError)
    # print("Center", info[0], "Area", info[1])
    cv2.imshow("Output", img)

    key_pressed = cv2.waitKey(1)
    if key_pressed in [ord('q'),27]: # 按键盘上的q或esc退出（在英文输入法下）
        break

me.land()
cv2.destroyAllWindows()