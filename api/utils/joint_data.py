"""
Joint data utilities for pose estimation models

This module provides utilities for working with joint position data
from various pose estimation models, including format conversions,
filtering, and saving to standard formats.
"""

import json
import numpy as np
import os
from typing import Dict, List, Any, Optional, Tuple, Union

# COCO-17 keypoint names for reference
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
]

# COCO-17 keypoint indices for different body parts
COCO_BODY_PARTS = {
    "face": [0, 1, 2, 3, 4],
    "upper_body": [5, 6, 7, 8, 9, 10],
    "lower_body": [11, 12, 13, 14, 15, 16],
    "left_arm": [5, 7, 9],
    "right_arm": [6, 8, 10],
    "left_leg": [11, 13, 15],
    "right_leg": [12, 14, 16],
    "torso": [5, 6, 11, 12]
}

# MediaPipe pose landmark indices mapping to COCO-17 keypoints
MEDIAPIPE_TO_COCO = {
    0: 0,    # nose
    2: 1,    # left_eye
    5: 2,    # right_eye
    7: 3,    # left_ear
    8: 4,    # right_ear
    11: 5,   # left_shoulder
    12: 6,   # right_shoulder
    13: 7,   # left_elbow
    14: 8,   # right_elbow
    15: 9,   # left_wrist
    16: 10,  # right_wrist
    23: 11,  # left_hip
    24: 12,  # right_hip
    25: 13,  # left_knee
    26: 14,  # right_knee
    27: 15,  # left_ankle
    28: 16   # right_ankle
}

class JointDataProcessor:
    """
    Utility class for processing and saving joint position data
    """
    
    @staticmethod
    def save_keypoints_to_json(
        keypoints: List[List[List[float]]],
        frame_indices: List[int],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        sample_rate: int = 1
    ) -> bool:
        """
        Save keypoint data to a JSON file
        
        Args:
            keypoints: List of keypoints for each frame [frames, keypoints, coords]
                       where coords is typically [x, y, z, confidence]
            frame_indices: List of frame indices corresponding to the keypoints
            output_path: Path to save the JSON file
            metadata: Optional metadata to include in the JSON
            sample_rate: Sampling rate (1 = every frame, 2 = every other frame, etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create data structure
            data = {
                "metadata": metadata or {},
                "frame_count": len(frame_indices),
                "keypoint_names": COCO_KEYPOINT_NAMES,
                "sample_rate": sample_rate,
                "frames": []
            }
            
            # Add keypoint data for each frame
            for i in range(len(frame_indices)):
                if i < len(keypoints):
                    frame_idx = frame_indices[i]
                    frame_keypoints = keypoints[i]
                    
                    # Format each keypoint as a dictionary
                    formatted_keypoints = []
                    for j, kp in enumerate(frame_keypoints):
                        # Handle both numpy arrays and lists
                        if isinstance(kp, np.ndarray):
                            kp = kp.tolist()
                            
                        formatted_keypoints.append({
                            "name": COCO_KEYPOINT_NAMES[j] if j < len(COCO_KEYPOINT_NAMES) else f"keypoint_{j}",
                            "position": {
                                "x": float(kp[0]),
                                "y": float(kp[1]),
                                "z": float(kp[2]) if len(kp) > 2 else 0.0
                            },
                            "confidence": float(kp[3]) if len(kp) > 3 else 1.0
                        })
                    
                    # Add frame data
                    data["frames"].append({
                        "index": int(frame_idx),
                        "keypoints": formatted_keypoints
                    })
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Saved keypoint data to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving keypoint data: {str(e)}")
            return False
    
    @staticmethod
    def apply_filter_to_keypoints(
        keypoints: List[List[List[float]]],
        filter_method: str,
        filter_window: int = 5,
        **filter_params
    ) -> List[List[List[float]]]:
        """
        Apply a filter to the keypoint trajectories
        
        Args:
            keypoints: List of keypoints for each frame [frames, keypoints, coords]
            filter_method: Name of the filter to apply
            filter_window: Window size for filter
            filter_params: Additional parameters for the filter
            
        Returns:
            Filtered keypoints
        """
        try:
            # Convert to numpy array for easier processing
            keypoints_array = np.array(keypoints)
            
            # If not enough frames for filtering, return original
            if len(keypoints_array) < filter_window:
                return keypoints
            
            # Apply filtering using the PoseFilter class
            from processors.noise_filters import PoseFilter
            filtered_keypoints = PoseFilter.apply_filter(
                keypoints_array, 
                filter_method, 
                filter_window,
                **filter_params
            )
            
            # Convert back to list and return
            return filtered_keypoints.tolist()
            
        except Exception as e:
            print(f"Error applying filter to keypoints: {str(e)}")
            return keypoints  # Return original if filtering fails
    
    @staticmethod
    def extract_coco_keypoints(
        landmarks: Any,  # Could be MediaPipe, Sapiens or other format
        source_type: str,
        frame_width: int,
        frame_height: int
    ) -> List[List[float]]:
        """
        Extract COCO-17 keypoint format from different landmark formats
        
        Args:
            landmarks: Landmarks in the source format
            source_type: Type of source ('mediapipe', 'sapiens', etc.)
            frame_width: Width of the video frame
            frame_height: Height of the video frame
            
        Returns:
            List of keypoints in COCO-17 format [x, y, z, confidence]
        """
        try:
            if source_type == 'mediapipe':
                # MediaPipe provides 33 keypoints, we map to COCO-17
                coco_keypoints = []
                for coco_idx in range(17):
                    mp_idx = next((k for k, v in MEDIAPIPE_TO_COCO.items() if v == coco_idx), None)
                    if mp_idx is not None and mp_idx < len(landmarks.landmark):
                        lm = landmarks.landmark[mp_idx]
                        # Convert normalized coordinates to pixel values
                        x = lm.x * frame_width
                        y = lm.y * frame_height
                        z = lm.z  # Keep as normalized depth
                        conf = lm.visibility
                        coco_keypoints.append([x, y, z, conf])
                    else:
                        # Add a placeholder for missing keypoints
                        coco_keypoints.append([0, 0, 0, 0])
                        
                return coco_keypoints
                
            elif source_type == 'sapiens':
                # Sapiens already outputs in COCO format, but we need to ensure correct dimensions
                coco_keypoints = []
                for i in range(17):
                    if i < len(landmarks):
                        kp = landmarks[i]
                        # Assuming landmarks are already in pixel coordinates
                        x, y = kp[0], kp[1]
                        z = kp[2] if len(kp) > 2 else 0.0
                        conf = kp[3] if len(kp) > 3 else 1.0
                        coco_keypoints.append([x, y, z, conf])
                    else:
                        coco_keypoints.append([0, 0, 0, 0])
                        
                return coco_keypoints
                
            elif source_type == '4dhumans':
                # Convert 4DHumans format to COCO-17
                # This is a placeholder - implement based on actual 4DHumans output format
                coco_keypoints = [[0, 0, 0, 0] for _ in range(17)]
                return coco_keypoints
                
            else:
                print(f"Unsupported landmark source type: {source_type}")
                return [[0, 0, 0, 0] for _ in range(17)]  # Return empty keypoints
                
        except Exception as e:
            print(f"Error extracting COCO keypoints: {str(e)}")
            return [[0, 0, 0, 0] for _ in range(17)]  # Return empty keypoints on error
    
    @staticmethod
    def load_keypoints_from_json(json_path: str) -> Tuple[List[List[List[float]]], List[int], Dict[str, Any]]:
        """
        Load keypoint data from a JSON file
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            Tuple containing:
            - List of keypoints for each frame [frames, keypoints, coords]
            - List of frame indices
            - Dictionary of metadata
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            keypoints = []
            frame_indices = []
            
            for frame in data.get('frames', []):
                frame_idx = frame['index']
                frame_keypoints = []
                
                for kp in frame['keypoints']:
                    pos = kp['position']
                    conf = kp.get('confidence', 1.0)
                    frame_keypoints.append([
                        pos['x'],
                        pos['y'],
                        pos.get('z', 0.0),
                        conf
                    ])
                
                keypoints.append(frame_keypoints)
                frame_indices.append(frame_idx)
            
            return keypoints, frame_indices, data.get('metadata', {})
            
        except Exception as e:
            print(f"Error loading keypoint data: {str(e)}")
            return [], [], {}
    
    @staticmethod
    def calculate_joint_angles(keypoints: List[List[float]]) -> Dict[str, float]:
        """
        Calculate joint angles from a single frame of keypoints
        
        Args:
            keypoints: List of keypoints in a single frame [keypoints, coords]
            
        Returns:
            Dictionary of joint angles in degrees
        """
        try:
            # Define joint triplets for angle calculation
            angle_definitions = {
                "left_elbow": [5, 7, 9],    # left_shoulder, left_elbow, left_wrist
                "right_elbow": [6, 8, 10],  # right_shoulder, right_elbow, right_wrist
                "left_shoulder": [11, 5, 7], # left_hip, left_shoulder, left_elbow
                "right_shoulder": [12, 6, 8], # right_hip, right_shoulder, right_elbow
                "left_knee": [11, 13, 15],  # left_hip, left_knee, left_ankle
                "right_knee": [12, 14, 16], # right_hip, right_knee, right_ankle
                "left_hip": [5, 11, 13],    # left_shoulder, left_hip, left_knee
                "right_hip": [6, 12, 14]    # right_shoulder, right_hip, right_knee
            }
            
            angles = {}
            
            for joint_name, (a_idx, b_idx, c_idx) in angle_definitions.items():
                # Get the three points forming the angle
                if a_idx < len(keypoints) and b_idx < len(keypoints) and c_idx < len(keypoints):
                    a = np.array(keypoints[a_idx][:2])  # Use only x,y coords
                    b = np.array(keypoints[b_idx][:2])
                    c = np.array(keypoints[c_idx][:2])
                    
                    # Calculate vectors
                    ba = a - b
                    bc = c - b
                    
                    # Calculate dot product and magnitudes
                    dot_product = np.dot(ba, bc)
                    magnitude_ba = np.linalg.norm(ba)
                    magnitude_bc = np.linalg.norm(bc)
                    
                    # Calculate angle in degrees
                    if magnitude_ba > 0 and magnitude_bc > 0:
                        cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
                        # Clamp cosine value to prevent domain errors
                        cosine_angle = max(min(cosine_angle, 1.0), -1.0)
                        angle_rad = np.arccos(cosine_angle)
                        angle_deg = np.degrees(angle_rad)
                        angles[joint_name] = float(angle_deg)
                    else:
                        angles[joint_name] = 0.0
                else:
                    angles[joint_name] = 0.0
            
            return angles
            
        except Exception as e:
            print(f"Error calculating joint angles: {str(e)}")
            return {}