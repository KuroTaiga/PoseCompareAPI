# Applies filter to a mp4 file.
# Input: filter name, filter window, base file
# Output: filtered mp4 file
import cv2
import os
import numpy as np
from scipy import signal
from config import Config

def apply_filter(video_path, pose_output, filter_name, filter_window):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(Config.RESULTS_FOLDER, f"{os.path.basename(video_path.split('.')[0])}_{filter_name}.mp4")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        match filter_name:
            case 'butterworth':
                b, a = signal.butter(4, filter_window, 'low')
            case 'chebyshev':
                b, a = signal.cheby1(4, 0.5, filter_window, 'low')
            case 'bessel':
                b, a = signal.bessel(4, filter_window, 'low')
            case _:
                raise ValueError(f"Unknown filter: {filter_name}")
    except Exception as e:
        raise ValueError(f"Error initializing video processing: {e}")