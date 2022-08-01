import os
import cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt


def create_tracker(tracker_type):
    if tracker_type == 'MIL':
        return cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        return cv2.TrackerKCF_create()
    elif tracker_type == 'CSRT':
        return cv2.TrackerCSRT_create()
    else:
        raise ValueError(f'Unknown tracker type: {tracker_type}')


def track(tracker_type, file_in, file_out, object_box):
    video_in = cv2.VideoCapture(file_in)

    _, frame = video_in.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = video_in.get(cv2.CAP_PROP_FPS)
    video_out = cv2.VideoWriter(file_out, fourcc, fps, (frame.shape[1], frame.shape[0]), True)

    tracker = create_tracker(tracker_type)
    tracker.init(frame, object_box)

    while True:
        ok, frame = video_in.read()
        if not ok:
            break

        ok, bbox = tracker.update(frame)
        if ok:
            x1, y1 = int(bbox[0]), int(bbox[1])
            width, height = int(bbox[2]), int(bbox[3])
            cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 0, 255), 2)
        video_out.write(frame)

    cv2.destroyAllWindows()
    video_out.release()


tracker_type = 'KCF'
track(
    tracker_type,
    file_in='/Users/ypolek/Downloads/football.mp4',
    file_out=f'/Users/ypolek/Downloads/football_{tracker_type}.avi',
    object_box=(789, 290, 19, 47)
)
# (460, 247, 35, 20) - chase
# (789, 290, 19, 47) - football
