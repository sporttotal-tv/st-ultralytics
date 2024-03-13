"""
Code inspired from: 
1. https://github.com/sporttotal-tv/cvat-api-sttv/blob/develop/scripts/sahi_yolov8_inference.py
2. https://github.com/sporttotal-tv/yolov5/blob/staging/yolov5/utils/sporttotal.py

"""
import os.path as osp
import argparse
import cv2
import pickle
from pathlib import Path
from typing import Optional, Union
from loguru import logger
from tqdm import tqdm

import torch
from ultralytics.utils.files import increment_path
from ultralytics import YOLO

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model

from st_commons.data.data_loader import VideoIterator

def convert_frame_predictiosn(results):
    player_frame_detections = {}
    ball_frame_detections = {}

    for bboxid, detection in enumerate(results.object_prediction_list):
        if detection.category.name == 'player':
            player_frame_detections[bboxid] = {
                                                'bbox': [int(v) for v in detection.bbox.to_xyxy()],
                                                'bbox_conf': round(detection.score.value, 5)
                                                }
        else:
            ball_frame_detections[bboxid] = {
                                                'bbox': [int(v) for v in detection.bbox.to_xyxy()],
                                                'bbox_conf': round(detection.score.value, 5)
                                            }
    return player_frame_detections, ball_frame_detections

def visualise_detections(frame_image, 
                         frame_id, 
                         player_frame_detections, 
                         ball_frame_detections,
                         out_dir = '/tmp',
                         debug = False):
    
    cv2.putText(frame_image, f"Frame {frame_id}: ", (frame_image.shape[0] // 3, 50), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 0, 255),
                3)

    bbox_color = (0,255,0) # green #compute_color_for_labels(int(bbox_id))

    for bbox_id in player_frame_detections:
        x1, y1, x2, y2 = player_frame_detections[bbox_id]["bbox"]
        cv2.rectangle(frame_image, (int(x1), int(y1)), (int(x2), int(y2)), bbox_color, 2)

        cv2.putText(frame_image, f"{bbox_id}", (int((x1 + x2) // 2) - 10, int((y1 + y2) // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if ball_frame_detections:
        bx, by = ball_frame_detections["bbox"]
        cv2.circle(frame_image, (int(bx), int(by)), 8, (0, 0, 255), 5)
    
    if debug:
        print("Dumping final image to disk")
        cv2.imwrite(str(Path(out_dir)/f"{frame_id}.jpg"), frame_image)

# def custom_visulisation(results, image_bgr):
    
#     object_prediction_list = results.object_prediction_list
#     boxes_list = []
#     clss_list = []
#     for ind, _ in enumerate(object_prediction_list):
#         clss = object_prediction_list[ind].category.name
#         boxes = (
#             object_prediction_list[ind].bbox.minx,
#             object_prediction_list[ind].bbox.miny,
#             object_prediction_list[ind].bbox.maxx,
#             object_prediction_list[ind].bbox.maxy,
#         )

#         boxes_list.append(boxes)
#         clss_list.append(clss)

#     # Create a copy of the original image to draw on
#     frame_copy = masked_image_bgr.copy()

#     for box, cls in zip(boxes_list, clss_list):
#         x1, y1, x2, y2 = box
#         cv2.rectangle(
#             frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 2
#         )
#         label = str(cls)
#         t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
#         cv2.rectangle(
#             frame_copy,
#             (int(x1), int(y1) - t_size[1] - 3),
#             (int(x1) + t_size[0], int(y1) + 3),
#             (56, 56, 255) if label == "person" else (56, 255, 56),
#             -1,
#         )
#         cv2.putText(
#             frame_copy,
#             label,
#             (int(x1), int(y1) - 2),
#             0,
#             0.6,
#             [255, 255, 255] if label == "person" else [0, 0, 0],
#             thickness=1,
#             lineType=cv2.LINE_AA,
#         )

#     frame_name = f"{frame_number:05d}_dets.jpg"
#     frame_path = save_dir / frame_name
#     cv2.imwrite(
#         str(frame_path),
#         frame_copy,
#     )
                
def detect_bboxes_from_video(video_path: str = None,
                             model_path: str = "yolov8n.pt",
                             model_type: Optional[str] = "yolov8",
                             start_time: Optional[Union[int,str]] = 0,
                             end_time: Optional[Union[int,str]] = None,
                             out_dir: Optional[str] = None,
                             out_path: Optional[str] = None,
                             court_mask_path: Optional[str] = None,
                             confidence_threshold: Optional[int] = 0.5,
                             device: Optional[str] = "cuda:0",
                             verbosity: int = 0,
                             debug: Optional[bool] = False,
                             **kwargs
):
    """
    Run object detection on images using YOLOv8 and SAHI.

    Args:
        weights_path: Model weights path.
        image_source: Image directory path.
        out_path: Path to the .bbox file.
        exist_ok: Overwrite existing files.
        debug: Save frames with boxes.
    """
    # Check source video
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Source video at '{video_path}' not found.")

    # Check model path
    if not Path(model_path).is_file():
        # download_yolov8s_model(model_path)
        raise FileNotFoundError(f"Model path '{model_path}' is not a file.")
    
    # Output setup
    if out_dir is None:
        out_dir = increment_path(Path(video_path).parent / "detection_results", exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"out_dir is: {out_dir}")
    
    if out_path is None:
        out_path = increment_path(Path(video_path).parent / "detection_results"/ (Path(video_path).stem+".bbox"), exist_ok=True)
    else:
        assert out_path.endswith(".bbox"), f"out_path must end with .bbox, got {out_path}"
    logger.info(f"out_path is: {out_path}")
    
    # Check court mask path
    use_court_mask = court_mask_path is not None
    if use_court_mask:
        assert osp.exists(court_mask_path), f"Cannot find court mask at {court_mask_path}"
        logger.info(f"Use court mask: {court_mask_path}")
        court_mask = cv2.imread(court_mask_path)
        
    model = YOLO(model_path)
    
    # Detect objects from classes 0 and 32 only
    # classes = [0, 32]
    # model.overrides["classes"] = classes

    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model=model,
        confidence_threshold=confidence_threshold,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    
    video = VideoIterator(video_path)
    video_basename = Path(video_path).stem
    
    video_writer = None
    player_detections = {}
    ball_detections = {}
    
    for frame in tqdm(video[start_time:end_time], desc=f"Player Detection for {video_basename} - "
                                               f"{0 if start_time is None else start_time}  to "
                                               f"{-1 if end_time is None else end_time}"):

        image_bgr = frame["image"]
        frame_id = frame["frame_id"]

        if use_court_mask:
            masked_image_bgr = cv2.bitwise_and(image_bgr, court_mask)
        else:
            masked_image_bgr = image_bgr
        print('verbosity', verbosity, debug)
        
            
        results = get_sliced_prediction(
            masked_image_bgr,
            detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        player_frame_detections, ball_frame_detections = convert_frame_predictiosn(results)

        player_detections[frame_id] = player_frame_detections
        ball_detections[frame_id] = ball_frame_detections
        
        if verbosity > 0:
            print("Visualisation")
            frame_overlay = visualise_detections(masked_image_bgr, 
                                                frame_id = frame_id, 
                                                player_frame_detections = player_frame_detections, 
                                                ball_frame_detections = ball_frame_detections,
                                                out_dir = out_dir,
                                                debug = debug)
            if out_path is not None:
                if video_writer is None:
                    video_out_path = str(out_path)+ ".mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(video_out_path, fourcc, video.frame_rate, (video.image_w, video.image_h))
                video_writer.write(frame_overlay)
            
            if verbosity > 2:  # interactive
                cv2.imshow("detector", frame_overlay)
                cv2.setWindowTitle("detector", video_basename)
                cv2.waitKey(1)

    det_data = {
                "players": player_detections,
                "ball": ball_detections,
                "debug":{
                    "frame_rate": video.frame_rate,
                    "image_height": video.image_height,
                    "image_width": video.image_width,
                    "model_name": Path(model_path).stem,
                }
                }

    if out_path is not None:
        pickle.dump(det_data, open(out_path, "wb"))
        logger.success(f"Saved bbox detections to {out_path}")
        if video_writer is not None:
            video_writer.release()
            logger.success(f"Saved bbox video to {video_out_path}")
    else:
        return det_data

    print("Inference with SAHI is done.")


# def detect_bboxes_from_images(image_dir: str = None,
#                             model_path: str = "yolov8n.pt",
#                             start_time: Optional[Union[int,str]] = 0,
#                             end_time:Optional[Union[int,str]] = None,
#                             out_dir: Optional[str] = None,
#                             out_path: Optional[str] = None,
#                             court_mask_path: Optional[str] = None,
#                             device: Optional[str] = "cuda:0",
#                             verbosity: int = 0,
#                             **kwargs
# ):
#     """
#     Run object detection on images using YOLOv8 and SAHI.

#     Args:
#         weights_path: Model weights path.
#         image_source: Image directory path.
#         exist_ok: Overwrite existing files.
#         debug: Save frames with boxes.
#     """
#     # Check source video or image path
#     if not Path(image_dir).is_dir() and not Path(video_path).is_file():
#         raise NotADirectoryError(f"Source path '{image_dir}'/'{video_path}' both are null.")

#     # Check model path
#     if not Path(model_path).is_file():
#         # download_yolov8s_model(model_path)
#         raise FileNotFoundError(f"Model path '{model_path}' is not a file.")
    
#     # Output setup
#     if out_dir is None:
#         out_dir = increment_path(Path(image_dir) / "results_sahi" / "exp", exist_ok=True)
#         out_dir.mkdir(parents=True, exist_ok=True)

#     # Check court mask path
#     use_court_mask = court_mask_path is not None
#     if use_court_mask:
#         assert osp.exists(court_mask_path), f"Cannot find court mask at {court_mask_path}"
#         logger.info(f"Use court mask: {court_mask_path}")
#         court_mask = cv2.imread(court_mask_path)
        
#     model = YOLO(model_path)
    
#     # Detect objects from classes 0 and 32 only
#     # classes = [0, 32]
#     # model.overrides["classes"] = classes

#     detection_model = AutoDetectionModel.from_pretrained(
#         model_type="yolov8",
#         model=model,
#         confidence_threshold=0.3,
#         device="cuda:0" if torch.cuda.is_available() else "cpu",
#     )

    
#     if image_dir:
#         image_files = list(Path(image_dir).rglob("*.[jp][pn]g"))
#         if not image_files:
#             raise FileNotFoundError(f"No image files found in: {image_dir}")

#     detections = {}
#     for img_path in image_files:
#         results = get_sliced_prediction(
#             str(img_path),
#             detection_model,
#             slice_height=512,
#             slice_width=512,
#             overlap_height_ratio=0.2,
#             overlap_width_ratio=0.2,
#         )

#         frame_number = int(img_path.stem)

#         object_prediction_list = [
#             res.to_coco_annotation().json for res in results.object_prediction_list
#         ]

#         detections[frame_number] = object_prediction_list