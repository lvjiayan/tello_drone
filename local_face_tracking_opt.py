import ctypes
import cv2
import numpy as np
import time
import os
import torch
import math
from ultralytics import YOLO
from cv2 import getTickCount, getTickFrequency
from PIL import Image, ImageDraw
from supervision import Detections
import sys

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

class Ai_tello:
    def __init__(self):
        # 加载 yolov5模型
        self.model = YOLO("model.pt")

        # print(self.model)
        # 置信度阈值
        # self.model.conf = 0.5

    def cameraProcess(self):

        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            loop_start = getTickCount()

            # 读取视频帧
            ret,frame = cap.read()
            if not ret:
                break

            # frame = cv2.resize(frame, (360, 240))
            frame = cv2.flip(frame, 1)

            # 转为RGB
            # img_cvt = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img_cvt = frame

            # 目标检测推理
            output = self.model(img_cvt)
            results = Detections.from_ultralytics(output[0])

            # annotated_frame = output[0].plot()

            # 获取所有人脸矩形框的边界
            # x_min = float(1000)
            # y_min = float(1000)
            # x_max = float(-1000)
            # y_max = float(-1000)

            for result in results.xyxy:
                # print(result)
                x_min = result[0]
                y_min = result[1]
                x_max = result[2]
                y_max = result[3]
                # x_min = min(1000, result[0])
                # y_min = min(1000, result[1])
                # x_max = max(-1000, result[2])
                # y_max = max(-1000, result[3])

                # print(img_cvt)
                # 确保边界不超出原图像的尺寸
                x_min = int(max(x_min, 0))
                y_min = int(max(y_min, 0))
                x_max = int(min(x_max, 640))
                y_max = int(min(y_max, 480))

                cv2.rectangle(img_cvt, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cx = (x_min + x_max) // 2
                cy = (y_min + y_max) // 2
                area = (x_max - x_min) * (y_max - y_min) 
                cv2.circle(img_cvt, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

            # 显示程序
            ticks = getTickCount() - loop_start #计算处理一帧图像所花费的时间。
            process_time = ticks / (getTickFrequency())  #将处理时间转换为秒数。
            FPS = int(1 / process_time) #FPS计算

            # 在图像左上角添加FPS文本
            fps_text = f"FPS: {FPS:.2f}"#构造显示帧率的文本字符串
            font = cv2.FONT_HERSHEY_SIMPLEX#选择字体类型
            font_scale = 1  #设置字体的缩放比例
            font_thickness = 2 #设置字体的粗细
            text_color = (0, 0, 255)  # 红色
            text_position = (10, 30)  # 左上角位置
        
            cv2.putText(img_cvt, fps_text, text_position, font, font_scale, text_color, font_thickness)

            cv2.imshow('demo', img_cvt)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()


    def flightControl(self):
        '''
        飞行控制进程调用函数
        '''

        # 启动另一个进程
        # Process(target=ai_tello.cameraProcess).start()
        ai_tello.cameraProcess()

if __name__ == '__main__':

    # 实例化
    ai_tello = Ai_tello()
    # 开启两个进程（防止飞机运动时，画面静止）
    ai_tello.flightControl()
    