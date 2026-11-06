import cv2 as cv
from ultralytics import YOLO
from rapidocr import RapidOCR
from datetime import datetime as dt
import torch, easyocr, csv, os, time
import numpy as np

def check_gpu():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_result_data_file(results, biggest_area, biggest_class_box, confidience_score, best_object_name):
    decimal_point = 2
    time_of_creation = dt.now()
    year = time_of_creation.year
    month = time_of_creation.month
    day = time_of_creation.day

    # records each objects detected
    class_ids = results.boxes.cls.int().tolist()
    class_boxes = [[round(coord, decimal_point) for coord in box] for box in results.boxes.xyxy.tolist()]
    class_names = []

    for ids in class_ids:
        class_names.append(results.names[ids])

    # check if file exist, if not then create one
    file_name = f'test_data/{day}{month}{year}_v2s_test_data.csv'
    with open(file_name, 'a+') as file:
        combined_ids_and_names = list(zip(class_ids,class_names))
        combined_names_and_boxes = list(zip(class_names,class_boxes))

        file.write(f'(class id + class name): {str(combined_ids_and_names)}\n')
        file.write(f'(class name + x1,y1, x2,y2): {str(combined_names_and_boxes)}')
        file.write(f''' 
best object: {best_object_name},
{best_object_name} area: {round(biggest_area,decimal_point)},
{best_object_name} box: {biggest_class_box}, 
confidience score of {best_object_name}: {confidience_score}\n
''')
        file.write('\n')

def screenshot(source, object, image_counter):
    time_of_creation = dt.now()
    year = time_of_creation.year
    month = time_of_creation.month
    day = time_of_creation.day
    closest_object_coordinates = object.get_coordinates()
    object_name = object.get_object_name()
    file_name = f'test_data/{image_counter}{day}{month}{year}_{object_name}_picture.png'
    if closest_object_coordinates != []:
        # frame[x1:x2, y1:y2]
        object_frame = source[closest_object_coordinates[1]:closest_object_coordinates[3], closest_object_coordinates[0]:closest_object_coordinates[2]] 
        cv.imwrite(file_name, object_frame)
        print('taken picture')
        image_counter+=1
    return image_counter

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

class create_class_information:
    def __init__(self,object_name, id, coordinates, confidience):
        self.name = object_name
        self.class_id = id
        self.coordinates = coordinates
        self.confidience_score = confidience

    def get_object_name(self):
        return self.name
    def get_class_id(self):
        return self.class_id

    def get_coordinates(self):
        return self.coordinates

    def get_confidience_score(self):
        return self.confidience_score

def YOLO_detection(model, target_classes, source, show_test=False):
    YOLO_results = model.predict(source=source, stream=True)
    best_class_id = 0
    confidience_score = 0
    green = (0, 255, 0)
    thickness = 2
    best_object_name = ''
    biggest_class_box = []
    biggest_area = 0

    for result in YOLO_results:
        class_ids = result.boxes.cls.int().tolist()
        confidience_scores = result.boxes.conf.tolist()
        class_boxes = result.boxes.xyxy.int().tolist()

        if show_test:
            print('class ids: ',class_ids)
            print('class boxes: ',class_boxes)
            print('confidience scores: ',confidience_scores)

        for target_index, class_id in enumerate(class_ids):
            if class_id in target_classes:
                x1, y1, x2, y2 = class_boxes[target_index]
                best_object_name = result.names[target_index]
                
                temp_area = (x2-x1) * (y2-y1)
                if temp_area > biggest_area:
                    biggest_area = temp_area
                    biggest_class_box = [x1,y1,x2,y2]

                    best_class_id = class_id
                    confidience_score = round(confidience_scores[target_index], 2)

                if show_test:
                    create_result_data_file(result, biggest_area, biggest_class_box, confidience_score, best_object_name)
                    print(f'{best_object_name} box: ',biggest_class_box)
                    cv.rectangle(source, (x1,y1), (x2,y2), green, thickness)
                #break

    closest_class_object = create_class_information(best_object_name, best_class_id, biggest_class_box, confidience_score)
    return closest_class_object



def main(test=False):
    # checking if device has a gpu or not
    device = check_gpu()

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
    image_counter = 1

    prev_frame_time = 0

    while True:
        ret, frame = cap.read()
        width = int(cap.get(3))
        height = int(cap.get(4))
        bottom_left = (0,height)
        key = cv.waitKey(1) 
        if not ret:
            break

        # object detection
        closets_object = YOLO_detection(YOLO_model, target_classes, frame, show_test=test)
        if key == ord('s') and test:
            image_counter = screenshot(frame, closets_object, image_counter)

        # RapidOCR, no gpu present
        if device == "cpu":
            pass

        # calculating fps
        prev_frame_time = show_fps(prev_frame_time, frame, bottom_left, font, font_scale, black, font_thickness)


        cv.imshow('v0.2.3', frame)

        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

main(test=True)
