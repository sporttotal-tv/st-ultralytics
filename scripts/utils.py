import cv2
from pathlib import Path


def convert_yolo_frame_predictions(dets, 
                                   confidence_threshold):
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
        if conf < confidence_threshold:
            continue
        
        if cls == 0:  # person
            player_frame_detections[idx] = {"bbox": bbox, "bbox_conf": conf}
        elif cls == 32:  # ball
            if conf > max_ball_conf:
                ball_frame_detections = {"bbox": bbox, "bbox_conf": conf}
                max_ball_conf = conf
                    
    return player_frame_detections, ball_frame_detections


def convert_sahi_frame_predictions(results):
    """
    Converts SAHI detection results into separate dictionaries for player and ball detections.

    Args:
        results (Prediction): The SAHI Prediction object containing detection results for a frame.

    Returns:
        tuple: A tuple containing two dictionaries:
            - player_frame_detections: A dictionary mapping each detected player's ID to its bounding box and confidence score.
            - ball_frame_detections: A dictionary mapping each detected ball's ID to its bounding box and confidence score.
    """
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