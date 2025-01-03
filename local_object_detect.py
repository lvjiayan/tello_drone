import cv2
import numpy as np
import torch

# 加载模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# model = torch.hub.load('./yolov5', 'custom', path='./weights/pose.pt',source='local')

# 从摄像头读取视频
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLOv5进行检测
    results = model(frame)

    # 显示检测结果
    results.render()  # 在frame上绘制检测框
    cv2.imshow('YOLOv5', frame)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()