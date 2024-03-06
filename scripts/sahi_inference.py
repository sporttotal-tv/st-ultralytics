"""
Code borrowed from: https://github.com/sporttotal-tv/cvat-api-sttv/blob/develop/scripts/sahi_yolov8_inference.py
"""
import os.path as osp
import argparse
import cv2
import pickle
from pathlib import Path
from typing import Optional, Union
from loguru import logger

import torch
from ultralytics.utils.files import increment_path
from ultralytics import YOLO

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model


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

    
def detect_bboxes_from_video(video_path: str = None,
                             model_path: str = "yolov8n.pt",
                             start_time: Optional[Union[int,str]] = 0,
                             end_time:Optional[Union[int,str]] = None,
                             out_dir: Optional[str] = None,
                             out_path: Optional[str] = None,
                             court_mask_path: Optional[str] = None,
                             device: Optional[str] = "cuda:0",
                             verbosity: int = 0,
                             **kwargs
):
    """
    Run object detection on images using YOLOv8 and SAHI.

    Args:
        weights_path: Model weights path.
        image_source: Image directory path.
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

    if out_path is None:
        out_path = increment_path(Path(video_path).parent / "detection_results"/ Path(video_path).stem + ".bbox", exist_ok=True)
    else:
        assert out_path.endswith(".bbox"), f"out_path must end with .bbox, got {out_path}"

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
        model_type="yolov8",
        model=model,
        confidence_threshold=0.3,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

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

        if debug:
            object_prediction_list = results.object_prediction_list
            boxes_list = []
            clss_list = []
            for ind, _ in enumerate(object_prediction_list):
                clss = object_prediction_list[ind].category.name
                boxes = (
                    object_prediction_list[ind].bbox.minx,
                    object_prediction_list[ind].bbox.miny,
                    object_prediction_list[ind].bbox.maxx,
                    object_prediction_list[ind].bbox.maxy,
                )

                boxes_list.append(boxes)
                clss_list.append(clss)

            # Create a copy of the original image to draw on
            frame_copy = masked_image_bgr.copy()

            for box, cls in zip(boxes_list, clss_list):
                x1, y1, x2, y2 = box
                cv2.rectangle(
                    frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 2
                )
                label = str(cls)
                t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
                cv2.rectangle(
                    frame_copy,
                    (int(x1), int(y1) - t_size[1] - 3),
                    (int(x1) + t_size[0], int(y1) + 3),
                    (56, 56, 255) if label == "person" else (56, 255, 56),
                    -1,
                )
                cv2.putText(
                    frame_copy,
                    label,
                    (int(x1), int(y1) - 2),
                    0,
                    0.6,
                    [255, 255, 255] if label == "person" else [0, 0, 0],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            frame_name = f"{frame_number:05d}_dets.jpg"
            frame_path = save_dir / frame_name
            cv2.imwrite(
                str(frame_path),
                frame_copy,
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    try:
        coco_out_path = f"{save_dir}/coco_results.pkl"
        print(f"Saving {coco_out_path}...")
        with open(coco_out_path, "wb") as f:
            pickle.dump(detections, f)
    except Exception as e:
        print(e)
        print(f"Could not save {coco_out_path}")

    print("Inference with SAHI is done.")

def detect_bboxes_from_images(image_dir: str = None,
                            model_path: str = "yolov8n.pt",
                            start_time: Optional[Union[int,str]] = 0,
                            end_time:Optional[Union[int,str]] = None,
                            out_dir: Optional[str] = None,
                            out_path: Optional[str] = None,
                            court_mask_path: Optional[str] = None,
                            device: Optional[str] = "cuda:0",
                            verbosity: int = 0,
                            **kwargs
):
    """
    Run object detection on images using YOLOv8 and SAHI.

    Args:
        weights_path: Model weights path.
        image_source: Image directory path.
        exist_ok: Overwrite existing files.
        debug: Save frames with boxes.
    """
    # Check source video or image path
    if not Path(image_dir).is_dir() and not Path(video_path).is_file():
        raise NotADirectoryError(f"Source path '{image_dir}'/'{video_path}' both are null.")

    # Check model path
    if not Path(model_path).is_file():
        # download_yolov8s_model(model_path)
        raise FileNotFoundError(f"Model path '{model_path}' is not a file.")
    
    # Output setup
    if out_dir is None:
        out_dir = increment_path(Path(image_dir) / "results_sahi" / "exp", exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

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
        model_type="yolov8",
        model=model,
        confidence_threshold=0.3,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    
    if image_dir:
        image_files = list(Path(image_dir).rglob("*.[jp][pn]g"))
        if not image_files:
            raise FileNotFoundError(f"No image files found in: {image_dir}")

    detections = {}
    for img_path in image_files:
        results = get_sliced_prediction(
            str(img_path),
            detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        frame_number = int(img_path.stem)

        object_prediction_list = [
            res.to_coco_annotation().json for res in results.object_prediction_list
        ]

        detections[frame_number] = object_prediction_list

        if debug:
            object_prediction_list = results.object_prediction_list
            boxes_list = []
            clss_list = []
            for ind, _ in enumerate(object_prediction_list):
                clss = object_prediction_list[ind].category.name
                boxes = (
                    object_prediction_list[ind].bbox.minx,
                    object_prediction_list[ind].bbox.miny,
                    object_prediction_list[ind].bbox.maxx,
                    object_prediction_list[ind].bbox.maxy,
                )

                boxes_list.append(boxes)
                clss_list.append(clss)

            frame = cv2.imread(str(img_path))

            # Create a copy of the original image to draw on
            frame_copy = frame.copy()

            for box, cls in zip(boxes_list, clss_list):
                x1, y1, x2, y2 = box
                cv2.rectangle(
                    frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 2
                )
                label = str(cls)
                t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
                cv2.rectangle(
                    frame_copy,
                    (int(x1), int(y1) - t_size[1] - 3),
                    (int(x1) + t_size[0], int(y1) + 3),
                    (56, 56, 255) if label == "person" else (56, 255, 56),
                    -1,
                )
                cv2.putText(
                    frame_copy,
                    label,
                    (int(x1), int(y1) - 2),
                    0,
                    0.6,
                    [255, 255, 255] if label == "person" else [0, 0, 0],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            frame_name = f"{frame_number:05d}_dets.jpg"
            frame_path = save_dir / frame_name
            cv2.imwrite(
                str(frame_path),
                frame_copy,
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    try:
        coco_out_path = f"{save_dir}/coco_results.pkl"
        print(f"Saving {coco_out_path}...")
        with open(coco_out_path, "wb") as f:
            pickle.dump(detections, f)
    except Exception as e:
        print(e)
        print(f"Could not save {coco_out_path}")

    print("Inference with SAHI is done.")

def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default="yolov8n.pt", help="initial weights path"
    )
    parser.add_argument(
        "--image-source", type=str, required=True, help="Images dir path"
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument("--debug", action="store_true", help="save frames with boxes")

    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)