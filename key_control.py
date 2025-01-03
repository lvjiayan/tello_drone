from djitellopy import tello
import key_press_module as kp
import time
import cv2

global img

kp.init()
me = tello.Tello()
me.connect()
print("----> 当前电池电量: ", me.get_battery())

me.streamon()

def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 30
    
    if kp.getKey("LEFT"): lr = -speed
    elif kp.getKey("RIGHT"): lr = speed

    if kp.getKey("UP"): fb = speed
    elif kp.getKey("DOWN"): fb = -speed
    
    if kp.getKey("w"): ud = speed
    elif kp.getKey("s"): ud = -speed    
        
    if kp.getKey("a"): yv = speed
    elif kp.getKey("d"): yv = -speed
    
    if kp.getKey("q"): me.land()
        
    if kp.getKey("r"): me.takeoff()
    
    if kp.getKey("f"):
        cv2.imwrite(f'Resources/Images/{time.time()}.jpg', img)
        time.sleep(0.3)
        
    return lr, fb, ud, yv

while True:
    val = getKeyboardInput()
    me.send_rc_control(val[0], val[1], val[2], val[3])
    
    img = me.get_frame_read().frame
    img = cv2.resize(img, (360, 240))
    cv2.imshow("Image", img)
    cv2.waitKey(1)
