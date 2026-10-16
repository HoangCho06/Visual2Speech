import cv2 as cv
import numpy as np
from ultralytics import YOLO
from rapidocr import RapidOCR
import torch, easyocr, csv, os, time, pyautogui

def create_result_data_file(results, biggest_area, biggest_class_box):
    # check if file exist, if not then create one
    file_name = 'test_data/data.csv'
    if os.path.exists(file_name):
        pass
    else:
        with open(file_name, 'w') as file:
            file.write('class id, class name, biggest area, biggest box\n')
            file.write('x1,y1, x2,y2\n\n')

    # records each objects detected
    class_ids = results.boxes.cls.int().tolist()
    class_boxes = [[round(coord, 2) for coord in box] for box in results.boxes.xyxy.tolist()]
    class_names = []

    for ids in class_ids:
        class_names.append(results.names[ids])

    with open('test_data/data.csv', 'a+') as file:
        combined_list = list(zip(class_ids,class_names))
        file.write(f'{str(combined_list)}, {round(biggest_area,2)}, {biggest_class_box} \n{str(class_boxes)}\n')
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

def YOLO_detection(model, target_classes, source, show_test=False):
    YOLO_results = model.predict(source=source, stream=True)
    class_id = 0
    green = (0, 255, 0)
    thickness = 2
    biggest_class_box = []
    biggest_area = 0

    for result in YOLO_results:
        class_ids = result.boxes.cls.int().tolist()
        class_boxes = result.boxes.xyxy.int().tolist()

        for class_id in class_ids:
            if class_id in target_classes:
                target_index = class_ids.index(class_id)
                x1, y1, x2, y2 = class_boxes[target_index]
                temp_area = (x2-x1) * (y2-y1)
                if temp_area > biggest_area:
                    biggest_area = temp_area
                    biggest_class_box = [x1,y1,x2,y2]

                if show_test:
                    create_result_data_file(result, biggest_area, biggest_class_box)
                    cv.rectangle(source, (x1,y1), (x2,y2), green, thickness)
                break

    return class_id, biggest_class_box



def main():

    YOLO_model = YOLO('model/yolo26n.pt')
    target_classes = [
        #0, # person 
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
        class_id, biggest_class_box = YOLO_detection(YOLO_model, target_classes, frame, show_test=True)

        # RapidOCR

        # calculating fps
        prev_frame_time = show_fps(prev_frame_time, frame, bottom_left, font, font_scale, black, font_thickness)


        cv.imshow('v0.1.4', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

main()
