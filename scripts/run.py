from fire import Fire
from st_ultralytics.inference import detect_bboxes_from_video

if __name__ == '__main__':
    Fire({"detect_bboxes_from_video": detect_bboxes_from_video})