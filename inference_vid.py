import os
import cv2
import random
from ultralytics import YOLO


def main():
    model = YOLO("visdrones/baseline/weights/best.pt") 
    model("Kyiv.mp4", save=True)
    # model("image.png", save=True)

if __name__ == "__main__":
    main()