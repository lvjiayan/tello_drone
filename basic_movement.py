from djitellopy import tello
from time import sleep

me = tello.Tello()
me.connect()
print("----> 剩余电池：", me.get_battery())

# 起飞
me.takeoff()
sleep(2)

# # 前进/后退
# me.send_rc_control(0, 50, 0, 0)
# sleep(3)

# # 前进/后退
# me.send_rc_control(0, -50, 0, 0)
# sleep(3)

# # 向右/向左
# me.send_rc_control(20, 0, 0, 0)
# sleep(3)

# # 向右/向左
# me.send_rc_control(-20, 0, 0, 0)
# sleep(3)

# 上升/下降
# me.send_rc_control(0, 0, 20, 0)
# sleep(2)

# # 旋转
# me.send_rc_control(0, 0, 0, 45)
# sleep(2)

# 停止动作
me.send_rc_control(0, 0, 0, 0)

# 着陆
me.land()