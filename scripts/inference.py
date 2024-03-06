import argparse
import torch
import cv2
import numpy as np
import pickle
from ultralytics import YOLO
from pprint import pprint

def get_args():
    parser = argparse.ArgumentParser("""Produce .bbox output for input video""")
    parser.add_argument("--video", type=str, help="path to input video", default="test_videos/first_goal_blue.mp4")
    parser.add_argument("--model", type=str, help="path to model", default="yolov8x.pt")
    # In COCO dataset, 0 is person and 32 is sport ball
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 32",
                        default=[0, 32])
    args = parser.parse_args()
    return args

def main(args):
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Format is described at https://github.com/sporttotal-tv/yolov5/
    output_dict = {
        "players": {},
        "ball": {},
        "debug": {
            "fps": fps,
            "image_h": height,
            "image_w": width,
            "model_name": args.model
        }
    }
    output_file = args.video.replace(".mp4", ".bbox")
    counter = 0
    while cap.isOpened():
        flag, image = cap.read()
        if flag:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            pickle.dump(output_dict, open(output_file, "wb"))
            break
        dets = model(image, imgsz=int(max(height, width)))
        bboxes = dets[0].boxes.xyxy.tolist()
        conf_scores = dets[0].boxes.conf.tolist()
        classes = dets[0].boxes.cls.tolist()
        output_dict["players"][counter] = {}
        max_ball_conf = -1
        for idx, (bbox, conf, cls) in enumerate(zip(bboxes, conf_scores, classes)):
            if cls not in args.classes:
                continue
            if cls == 0:  # person
                output_dict["players"][counter][idx] = {"bbox": bbox, "bbox_conf": conf}
            elif cls == 32:  # ball
                if conf > max_ball_conf:
                    output_dict["ball"][counter] = {"bbox": bbox, "bbox_conf": conf}
                    max_ball_conf = conf
        counter += 1
        
if __name__ == "__main__":
    args = get_args()
    main(args)