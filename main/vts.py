import cv2 as cv
from ultralytics import YOLO
import torch, easyocr, csv, os, time
import numpy as np

def create_test_data_file(results):
    # check if file exist, if not then create one
    file_name = 'data.csv'
    if os.path.exists(file_name):
        pass
    else:
        with open(file_name, 'w') as file:
            file.write('(class id, class name)\n')

    # records each objects detected
    class_ids = results.boxes.cls.int().tolist()
    class_names = []
    for ids in class_ids:
        class_names.append(results.names[ids])

    with open('data.csv', 'a+') as file:
        combined_list = list(zip(class_ids,class_names))
        file.write(str(combined_list))
        file.write('\n')

def show_fps(prev, source, position, font, font_scale, colour, font_thickness):
        new = time.time()
        fps = int(1/(new-prev))
        prev = new

        cv.putText(
            source,
            str(fps),
            position,
            font,
            font_scale,
            colour,
            font_thickness,
            cv.LINE_AA
            )

        return prev

def main():

    YOLO_model = YOLO('visual_to_speech/model/yolov8n.pt')
    target_classes = [
        0, # person 
        2, # car
        7, # truck 
        11, # stop sign
        67, # cell phone
        73, # laptop screen
        76, # keyboard
        77  # book covers
    ]

    camera_model = 0
    cap = cv.VideoCapture(camera_model)
    font = cv.FONT_HERSHEY_COMPLEX
    black = (0,0,0)
    font_scale = 1
    font_thickness = 1

    prev_frame_time = 0

    while True:
        ret, frame = cap.read()
        width = int(cap.get(3))
        height = int(cap.get(4))
        bottom_left = (0,height)

        # object detection
        YOLO_results = YOLO_model.predict(source=frame, stream=True)
        for result in YOLO_results:
            class_ids = result.boxes.cls.int().tolist()
            create_test_data_file(result)
            frame = result.plot()
            break

        # calculating fps
        prev_frame_time = show_fps(prev_frame_time, frame, bottom_left, font, font_scale, black, font_thickness)


        cv.imshow('v0.1.1', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

main()
