"""
MediaPipe pose estimation model processor

This module provides a processor for the MediaPipe pose estimation model,
using shared components for filtering, interpolation, and visualization.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from utils.joint_data import JointDataProcessor
from processors.noise_filters import PoseFilter
from processors.interpolation import PoseInterpolation
from utils.visualization import PoseVisualizer
from utils.joint_data import COCO_KEYPOINT_NAMES

class MediaPipeProcessor:
    """Processor for MediaPipe pose estimation model"""
    
    def __init__(self):
        """Initialize MediaPipe pose model"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def process_video(self, input_path, output_path, method='original', 
                      filter_window=5, output_json_path=None, sample_rate=1,
                      max_frames=10000, radius=4, thickness=2):
        """
        Process a video with MediaPipe pose model
        
        Args:
            input_path (str): Path to input video
            output_path (str): Path to save processed video
            method (str): Filtering or interpolation method to apply
            filter_window (int): Window size for filtering
            output_json_path (str, optional): Path to save joint data as JSON
            sample_rate (int): Save every Nth frame to JSON
            max_frames (int): Maximum number of frames to process
            radius (int): Radius of keypoint circles
            thickness (int): Line thickness for skeleton
            
        Returns:
            dict: Processing results with paths to output files
        """
        print(f"Processing video with MediaPipe ({method})")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_path}")
            return None
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Storage for landmarks and keypoints
        all_landmarks = []
        all_keypoints = []
        frame_indices = []
        
        # Process frames
        frame_count = 0
        
        print("Reading frames and detecting landmarks...")
        while cap.isOpened() and frame_count < min(max_frames, total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Only process every nth frame according to sample_rate
            if frame_count % sample_rate == 0 or method == 'original':
                print(f"\rProcessing frame {frame_count}/{min(max_frames, total_frames)}", end="")
                    
                # Store frame index
                frame_indices.append(frame_count)
                
                # Process with MediaPipe
                results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if results.pose_landmarks:
                    # Extract landmarks for filtering/interpolation
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                    all_landmarks.append(landmarks)
                    
                    # Extract COCO format keypoints for JSON
                    coco_keypoints = JointDataProcessor.extract_coco_keypoints(
                        results.pose_landmarks, 
                        'mediapipe',
                        width,
                        height
                    )
                    all_keypoints.append(coco_keypoints)
                else:
                    # Add placeholder for missing landmarks/keypoints
                    all_landmarks.append([])
                    all_keypoints.append([[0, 0, 0, 0] for _ in range(17)])  # Empty COCO keypoints
        
        print(f"\nLandmark detection completed for {len(all_landmarks)} frames")
        cap.release()
        
        # Convert to numpy arrays
        all_landmarks = np.array(all_landmarks)
        all_keypoints = np.array(all_keypoints)
        
        # Determine whether to use filter or interpolation based on method
        filter_methods = ['original', 'butterworth', 'kalman', 'wiener', 'chebyshev', 'bessel']
        interpolation_methods = ['linear', 'spline', 'bilinear', 'kriging']
        
        # Process landmarks and keypoints
        if len(all_landmarks) > 0:
            if method in filter_methods:
                # Apply noise filter
                if method != 'original':
                    print(f"Applying {method} filter...")
                    processed_landmarks = PoseFilter.apply_filter(
                        all_landmarks, 
                        method, 
                        filter_window
                    )
                    
                    processed_keypoints = PoseFilter.apply_filter(
                        all_keypoints,
                        method,
                        filter_window
                    )
                else:
                    processed_landmarks = all_landmarks
                    processed_keypoints = all_keypoints
            
            elif method in interpolation_methods:
                # Apply interpolation
                print(f"Applying {method} interpolation...")
                processed_landmarks = PoseInterpolation.apply_interpolation(
                    all_landmarks,
                    method
                )
                
                processed_keypoints = PoseInterpolation.apply_interpolation(
                    all_keypoints,
                    method
                )
            
            else:
                # Unknown method, use original
                print(f"Unknown method '{method}', using original data")
                processed_landmarks = all_landmarks
                processed_keypoints = all_keypoints
        else:
            # No detections
            print("No pose landmarks detected in video")
            return None
        
        # Save keypoints to JSON if requested
        if output_json_path and len(all_keypoints) > 0:
            print(f"Saving keypoints to {output_json_path}")
            
            metadata = {
                "model": "mediapipe",
                "method": method,
                "filter_window": filter_window,
                "video_path": input_path,
                "frame_width": width,
                "frame_height": height,
                "fps": fps,
                "total_frames": total_frames
            }
            
            JointDataProcessor.save_keypoints_to_json(
                processed_keypoints,
                frame_indices,
                output_json_path,
                metadata=metadata,
                sample_rate=sample_rate
            )
        
        # Generate output video with overlaid poses
        print("Generating output video with overlaid poses...")
        
        # Create output video using the visualization utility
        # First convert landmarks to a format compatible with the visualizer
        keypoints_data = {
            "metadata": {
                "model": "mediapipe",
                "method": method,
                "filter_window": filter_window
            },
            "frame_count": len(frame_indices),
            "keypoint_names": COCO_KEYPOINT_NAMES,
            "sample_rate": sample_rate,
            "frames": []
        }
        
        for i, frame_idx in enumerate(frame_indices):
            if i < len(processed_keypoints) and len(processed_keypoints[i]) > 0:
                # Format keypoints for the visualizer
                keypoints_list = []
                for j, kp in enumerate(processed_keypoints[i]):
                    keypoints_list.append({
                        "name": COCO_KEYPOINT_NAMES[j] if j < len(COCO_KEYPOINT_NAMES) else f"keypoint_{j}",
                        "position": {
                            "x": float(kp[0]),
                            "y": float(kp[1]),
                            "z": float(kp[2]) if len(kp) > 2 else 0.0
                        },
                        "confidence": float(kp[3]) if len(kp) > 3 else 1.0
                    })
                
                keypoints_data["frames"].append({
                    "index": int(frame_idx),
                    "keypoints": keypoints_list
                })
        
        # Create the output video
        success = PoseVisualizer.create_pose_video(
            input_path,
            output_path,
            keypoints_data,
            confidence_threshold=0.3,
            keypoint_radius=radius,
            line_thickness=thickness,
            max_frames=max_frames
        )
        
        if success:
            print(f"Processing complete. Output saved to {output_path}")
            if output_json_path:
                print(f"Keypoint data saved to {output_json_path}")
            
            # Return results
            return {
                "video": output_path,
                "json": output_json_path if output_json_path else None
            }
        else:
            print("Failed to create output video")
            return None