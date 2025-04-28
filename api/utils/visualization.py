"""
Visualization utilities for pose data

This module provides functions for visualizing pose data, 
including creating videos with pose overlays.
"""

import cv2
import numpy as np
import json
import os
from utils.joint_data import COCO_KEYPOINT_NAMES

# Default colors for COCO keypoints (BGR format)
DEFAULT_KEYPOINT_COLORS = [
    (0, 255, 255),    # nose - yellow
    (0, 255, 0),      # left_eye - green
    (0, 255, 0),      # right_eye - green
    (0, 255, 0),      # left_ear - green
    (0, 255, 0),      # right_ear - green
    (255, 0, 0),      # left_shoulder - blue
    (0, 0, 255),      # right_shoulder - red
    (255, 0, 0),      # left_elbow - blue
    (0, 0, 255),      # right_elbow - red
    (255, 0, 0),      # left_wrist - blue
    (0, 0, 255),      # right_wrist - red
    (255, 0, 0),      # left_hip - blue
    (0, 0, 255),      # right_hip - red
    (255, 0, 0),      # left_knee - blue
    (0, 0, 255),      # right_knee - red
    (255, 0, 0),      # left_ankle - blue
    (0, 0, 255)       # right_ankle - red
]

# Default connections for skeleton visualization
DEFAULT_SKELETON_CONNECTIONS = [
    (0, 1),  # nose to left_eye
    (0, 2),  # nose to right_eye
    (1, 3),  # left_eye to left_ear
    (2, 4),  # right_eye to right_ear
    (5, 6),  # left_shoulder to right_shoulder
    (5, 7),  # left_shoulder to left_elbow
    (7, 9),  # left_elbow to left_wrist
    (6, 8),  # right_shoulder to right_elbow
    (8, 10), # right_elbow to right_wrist
    (5, 11), # left_shoulder to left_hip
    (6, 12), # right_shoulder to right_hip
    (11, 12), # left_hip to right_hip
    (11, 13), # left_hip to left_knee
    (13, 15), # left_knee to left_ankle
    (12, 14), # right_hip to right_knee
    (14, 16)  # right_knee to right_ankle
]

class PoseVisualizer:
    """
    Class for visualizing pose data in various formats
    """
    
    @staticmethod
    def create_pose_video(
        input_video_path,
        output_video_path,
        keypoints_data,
        confidence_threshold=0.3,
        keypoint_radius=4,
        line_thickness=2,
        keypoint_colors=None,
        skeleton_connections=None,
        max_frames=None,
        resize_dims=None
    ):
        """
        Create a video with pose keypoints overlaid
        
        Args:
            input_video_path (str): Path to input video
            output_video_path (str): Path to save output video
            keypoints_data (str or dict): Path to keypoints JSON file or loaded keypoints data
            confidence_threshold (float): Threshold for displaying keypoints
            keypoint_radius (int): Radius of keypoint circles
            line_thickness (int): Thickness of skeleton lines
            keypoint_colors (list): Colors for each keypoint (BGR format)
            skeleton_connections (list): Connections between keypoints for skeleton
            max_frames (int): Maximum number of frames to process
            resize_dims (tuple): Dimensions to resize video frames (width, height)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load keypoints data
            if isinstance(keypoints_data, str):
                with open(keypoints_data, 'r') as f:
                    keypoints_data = json.load(f)
            
            # Set defaults if not provided
            if keypoint_colors is None:
                keypoint_colors = DEFAULT_KEYPOINT_COLORS
            
            if skeleton_connections is None:
                skeleton_connections = DEFAULT_SKELETON_CONNECTIONS
            
            # Open input video
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file {input_video_path}")
                return False
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Apply resize if specified
            if resize_dims is not None:
                width, height = resize_dims
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # Extract frame data from keypoints
            frame_data = {}
            for frame in keypoints_data.get('frames', []):
                frame_data[frame['index']] = frame['keypoints']
            
            # Limit the number of frames if specified
            if max_frames is None:
                max_frames = total_frames
            else:
                max_frames = min(max_frames, total_frames)
            
            # Process frames
            frame_count = 0
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Resize if needed
                if resize_dims is not None:
                    frame = cv2.resize(frame, resize_dims)
                
                # Draw pose if we have data for this frame
                if frame_count in frame_data:
                    frame = PoseVisualizer.draw_pose_on_frame(
                        frame,
                        frame_data[frame_count],
                        confidence_threshold,
                        keypoint_radius,
                        line_thickness,
                        keypoint_colors,
                        skeleton_connections
                    )
                
                # Write frame to output video
                out.write(frame)
            
            # Release resources
            cap.release()
            out.release()
            
            print(f"Pose visualization complete. Output saved to {output_video_path}")
            return True
            
        except Exception as e:
            print(f"Error creating pose video: {str(e)}")
            return False
    
    @staticmethod
    def draw_pose_on_frame(
        frame,
        keypoints,
        confidence_threshold=0.3,
        keypoint_radius=4,
        line_thickness=2,
        keypoint_colors=None,
        skeleton_connections=None
    ):
        """
        Draw pose keypoints and skeleton on a single frame
        
        Args:
            frame (numpy.ndarray): Input frame
            keypoints (list): List of keypoint data
            confidence_threshold (float): Threshold for displaying keypoints
            keypoint_radius (int): Radius of keypoint circles
            line_thickness (int): Thickness of skeleton lines
            keypoint_colors (list): Colors for each keypoint (BGR format)
            skeleton_connections (list): Connections between keypoints for skeleton
            
        Returns:
            numpy.ndarray: Frame with pose overlay
        """
        # Create a copy of the frame to avoid modifying the original
        result_frame = frame.copy()
        
        # Set defaults if not provided
        if keypoint_colors is None:
            keypoint_colors = DEFAULT_KEYPOINT_COLORS
        
        if skeleton_connections is None:
            skeleton_connections = DEFAULT_SKELETON_CONNECTIONS
        
        # Extract keypoint coordinates and confidence values
        coords = []
        confs = []
        
        for i, kp in enumerate(keypoints):
            # Handle both array format and dictionary format
            if isinstance(kp, dict):
                # Dictionary format from JSON
                if 'position' in kp and 'confidence' in kp:
                    x = int(kp['position']['x'])
                    y = int(kp['position']['y'])
                    conf = float(kp['confidence'])
                    coords.append((x, y))
                    confs.append(conf)
            elif isinstance(kp, list):
                # Array format [x, y, z, conf]
                if len(kp) >= 4:
                    x = int(kp[0])
                    y = int(kp[1])
                    conf = float(kp[3])
                    coords.append((x, y))
                    confs.append(conf)
        
        # Draw skeleton connections first (so they appear behind keypoints)
        for connection in skeleton_connections:
            i, j = connection
            if i < len(coords) and j < len(coords) and confs[i] > confidence_threshold and confs[j] > confidence_threshold:
                # Get the average color of the two keypoints for the connection
                if i < len(keypoint_colors) and j < len(keypoint_colors):
                    color1 = np.array(keypoint_colors[i])
                    color2 = np.array(keypoint_colors[j])
                    color = tuple(map(int, (color1 + color2) / 2))
                else:
                    color = (0, 255, 0)  # Default green if colors not available
                
                # Draw the line
                cv2.line(result_frame, coords[i], coords[j], color, thickness=line_thickness)
        
        # Draw keypoints
        for i, ((x, y), conf) in enumerate(zip(coords, confs)):
            if conf > confidence_threshold:
                color = keypoint_colors[i] if i < len(keypoint_colors) else (0, 255, 0)
                cv2.circle(result_frame, (x, y), keypoint_radius, color, -1)
        
        return result_frame
    
    @staticmethod
    def create_video_from_model_output(
        input_video_path,
        output_video_path,
        model_output,
        model_type='mediapipe',
        confidence_threshold=0.3,
        max_frames=None
    ):
        """
        Create a video with pose overlay from a model's direct output format
        
        Args:
            input_video_path (str): Path to input video
            output_video_path (str): Path to save output video
            model_output: Model-specific pose estimation output
            model_type (str): Type of model ('mediapipe', 'sapiens', etc.)
            confidence_threshold (float): Threshold for keypoint visibility
            max_frames (int): Maximum number of frames to process
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Process model output based on the model type
            if model_type == 'mediapipe':
                # For MediaPipe, convert landmark format
                from utils.joint_data import JointDataProcessor
                
                # Get video dimensions for conversion
                cap = cv2.VideoCapture(input_video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                # Create keypoints data in the standard format
                keypoints_data = {
                    'frames': []
                }
                
                for frame_idx, landmarks in enumerate(model_output):
                    if landmarks:
                        # Extract COCO format keypoints
                        coco_keypoints = JointDataProcessor.extract_coco_keypoints(
                            landmarks, 
                            'mediapipe',
                            width,
                            height
                        )
                        
                        # Format for visualization
                        keypoints_data['frames'].append({
                            'index': frame_idx + 1,  # 1-based indexing
                            'keypoints': coco_keypoints
                        })
                
                # Create the video
                return PoseVisualizer.create_pose_video(
                    input_video_path,
                    output_video_path,
                    keypoints_data,
                    confidence_threshold=confidence_threshold,
                    max_frames=max_frames
                )
            
            elif model_type == 'sapiens':
                # For Sapiens, use its native COCO format or convert if needed
                # (Implementation will depend on the specific format of Sapiens output)
                print("Sapiens model output processing not implemented yet")
                return False
            
            else:
                print(f"Unsupported model type: {model_type}")
                return False
                
        except Exception as e:
            print(f"Error creating video from model output: {str(e)}")
            return False