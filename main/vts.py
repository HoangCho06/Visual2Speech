import cv2 as cv
import ultralytics as YOLO
import torch, easyocr
import numpy as np


def main():
    camera_model = 0
    cap = cv.VideoCapture(camera_model)

    fps = cap.get(cv.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()


        cv.imshow('v0.1.0', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

main()
