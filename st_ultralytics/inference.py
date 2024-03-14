"""
This script performs object detection on a pano video using a custom-trained YOLOv8 model.
It detects players and the ball in each frame and saves the detections to a .bbox file.

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

from st_ultralytics.utils import convert_yolo_frame_predictions, convert_sahi_frame_predictions, visualise_detections

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model

from st_commons.data.data_loader import VideoIterator
from st_commons.tools.general import prepare_cfg
           
def detect_bboxes_from_video(video_path: str = None,
                             model_path: str = None,
                             model_type: Optional[str] = "yolov8",
                             start_time: Optional[Union[int,str]] = 0,
                             end_time: Optional[Union[int,str]] = None,
                             out_dir: Optional[str] = None,
                             out_path: Optional[str] = None,
                             court_mask_path: Optional[str] = None,
                             device: Optional[str] = "cuda:0",
                             verbosity: int = 0,
                             debug: Optional[bool] = False,
                             sahi_inference: Optional[bool] = False,
                             **kwargs
):
    """
    Run object detection on images using YOLOv8 and SAHI.

    Args:
        video_path: Path to the input video file.
        model_path: Path to the custom-trained model weights (default: "yolov8n.pt").
        model_type: Type of the model (default: "yolov8").
        start_time: Start time of the video segment to process (default: 0).
        end_time: End time of the video segment to process (default: None).
        out_dir: Output directory for saving the detection results (default: None).
        out_path: Path to save the .bbox file (default: None).
        court_mask_path: Path to the court mask image (default: None).
        confidence_threshold: Confidence threshold for object detection (default: 0.5).
        device: Device to run the inference on (default: "cuda:0").
        verbosity: Verbosity level (default: 0).
        debug: Flag to save the visualized frames to disk (default: False).
        **kwargs: Additional keyword arguments.

    Returns:
        dict: Detection data containing player and ball detections.
    """
    # Check source video
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Source video at '{video_path}' not found.")

    # Read config
    config = prepare_cfg()   
    
    # Check model path
    if model_path is None:
        if sahi_inference:
            model_path = config.sahi.model_path
        else:
            if model_type == "yolov8":               
                model_path = config.yolov8.model_path
        
    if not Path(model_path).is_file():
        # download_yolov8s_model(model_path)
        raise FileNotFoundError(f"Model path '{model_path}' is not a file.")
    
    # Output setup
    if out_dir is None:
        out_dir = increment_path(Path(video_path).parent / "sahi_detection_results", exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"out_dir is: {out_dir}")
    
    if out_path is None:
        if sahi_inference: exp_name = 'sahi_detection_results'
        else: exp_name = 'detection_results'
        out_path = increment_path(Path(video_path).parent / exp_name / (Path(video_path).stem+".bbox"), exist_ok=True)
    else:
        assert out_path.endswith(".bbox"), f"out_path must end with .bbox, got {out_path}"
    logger.info(f"out_path is: {out_path}")
    
    # Check court mask path
    use_court_mask = court_mask_path is not None
    if use_court_mask:
        assert osp.exists(court_mask_path), f"Cannot find court mask at {court_mask_path}"
        logger.info(f"Use court mask: {court_mask_path}")
        court_mask = cv2.imread(court_mask_path)
        
    video = VideoIterator(video_path)
    video_basename = Path(video_path).stem
    video_writer = None
     
    model = YOLO(model_path)
    
    # Detect objects from classes 0 and 32 only
    # classes = [0, 32]
    # model.overrides["classes"] = classes

    if sahi_inference:
        logger.info(f"Sahi based inferencing is enabled")
        model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model=model,
            confidence_threshold=config.sahi.confidence_threshold,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
    
    player_detections = {}
    ball_detections = {}
    
    for frame in tqdm(video[start_time:end_time], desc=f"Player Detection for {video_basename} - "
                                               f"{0 if start_time is None else start_time}  to "
                                               f"{-1 if end_time is None else end_time}"):

        image_bgr = frame["image"]
        frame_id = frame["frame_id"]
        height, width, _ = image_bgr.shape
        
        if use_court_mask:
            masked_image_bgr = cv2.bitwise_and(image_bgr, court_mask)
        else:
            masked_image_bgr = image_bgr
        
        if sahi_inference:
            results = get_sliced_prediction(
                masked_image_bgr,
                model,
                slice_height=config.sahi.slice_height,
                slice_width=config.sahi.slice_width,
                overlap_height_ratio=config.sahi.overlap_height_ratio,
                overlap_width_ratio=config.sahi.overlap_width_ratio,
            )
            player_frame_detections, ball_frame_detections = convert_sahi_frame_predictions(results)
        else:
            results = model(masked_image_bgr, imgsz=int(max(height, width)))
            player_frame_detections, ball_frame_detections = convert_yolo_frame_predictions(results, 
                                                                                            config.yolov8.confidence_threshold)            
            
        player_detections[frame_id] = player_frame_detections
        ball_detections[frame_id] = ball_frame_detections
        
        if verbosity > 0:
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