from djitellopy import tello
import cv2
from time import sleep

me = tello.Tello()
me.connect()
print("剩余电池电量：", me.get_battery())

# 起飞
# me.takeoff()
# sleep(2)

me.streamon()

while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img, (360, 240))
    cv2.imshow("Image", img)
    cv2.waitKey(1)
