from djitellopy import tello
import cv2
import numpy as np
import time
import logging

tello.Tello.LOGGER.setLevel(logging.WARNING)

window_width, window_height = [360, 240]
fbRange = [6300, 6600]
pid = [0.4, 0.4, 0]
pError = 0

def findFace(img):
    faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
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

last_time = time.clock()

while True:
    # _, img = cap.read()
    cur_time = time.clock()
    if cur_time - last_time > 1.0:
        print("当前电池电量：", me.get_battery())
        last_time = cur_time
    img = me.get_frame_read().frame
    img = cv2.resize(img, (window_width, window_height))
    img, info = findFace(img)
    pError = trackFace(me, info, window_width, pid, pError)
    # print("Center", info[0], "Area", info[1])
    cv2.imshow("Output", img)

    key_pressed = cv2.waitKey(20)
    if key_pressed in [ord('q'),27]: # 按键盘上的q或esc退出（在英文输入法下）
        me.land()
        break