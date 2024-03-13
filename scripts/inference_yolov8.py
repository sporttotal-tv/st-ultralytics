"""
Code inspired from: 
1. https://github.com/sporttotal-tv/cvat-api-sttv/blob/develop/scripts/sahi_yolov8_inference.py
2. https://github.com/sporttotal-tv/yolov5/blob/staging/yolov5/utils/sporttotal.py

"""
import os.path as osp
import cv2
import pickle
from pathlib import Path
from typing import Optional, Union
from loguru import logger
from tqdm import tqdm

import torch
from ultralytics.utils.files import increment_path
from ultralytics import YOLO

from st_commons.data.data_loader import VideoIterator

def convert_frame_predictiosn(dets):
    """
    Convert frame predictions to player and ball detections.

    Args:
        dets: Frame detections from the YOLOv8 model.

    Returns:
        tuple: Player frame detections and ball frame detections.
    """
    player_frame_detections = {}
    ball_frame_detections = {}

    bboxes = dets[0].boxes.xyxy.tolist()
    conf_scores = dets[0].boxes.conf.tolist()
    classes = dets[0].boxes.cls.tolist()
    
    max_ball_conf = -1        
    for idx, (bbox, conf, cls) in enumerate(zip(bboxes, conf_scores, classes)):
        if cls == 0:  # person
            player_frame_detections[idx] = {"bbox": bbox, "bbox_conf": conf}
        elif cls == 32:  # ball
            if conf > max_ball_conf:
                ball_frame_detections = {"bbox": bbox, "bbox_conf": conf}
                max_ball_conf = conf
                    
    return player_frame_detections, ball_frame_detections

def visualise_detections(frame_image, 
                         frame_id, 
                         player_frame_detections, 
                         ball_frame_detections,
                         out_dir = '/tmp',
                         debug = False):
    """
    Visualize player and ball detections on the frame image.

    Args:
        frame_image: The frame image.
        frame_id: The frame ID.
        player_frame_detections: Player detections for the frame.
        ball_frame_detections: Ball detections for the frame.
        out_dir: Output directory for saving the visualized frames (default: '/tmp').
        debug: Flag to save the visualized frames to disk (default: False).

    Returns:
        numpy.ndarray: The frame image with visualized detections.
    """
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
        bx, by = ball_frame_detections["bbox"][:2]
        cv2.circle(frame_image, (int(bx), int(by)), 8, (0, 0, 255), 5)
    
    if debug:
        print("Dumping final image to disk")
        cv2.imwrite(str(Path(out_dir)/f"{frame_id}.jpg"), frame_image)
    
    return frame_image
                
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
        height, width, _ = image_bgr.shape
        
        if use_court_mask:
            masked_image_bgr = cv2.bitwise_and(image_bgr, court_mask)
        else:
            masked_image_bgr = image_bgr
        
        dets = model(masked_image_bgr, imgsz=int(max(height, width)))
                    
        # results = get_sliced_prediction(
        #     masked_image_bgr,
        #     detection_model,
        #     slice_height=512,
        #     slice_width=512,
        #     overlap_height_ratio=0.2,
        #     overlap_width_ratio=0.2,
        # )

        player_frame_detections, ball_frame_detections = convert_frame_predictiosn(dets)

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