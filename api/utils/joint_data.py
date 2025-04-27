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
            
            # Add keypoint data for each frame, applying sampling rate
            for i in range(0, len(frame_indices), sample_rate):
                frame_idx = frame_indices[i]
                frame_keypoints = keypoints[i]
                
                # Format each keypoint as a dictionary
                formatted_keypoints = []
                for j, kp in enumerate(frame_keypoints):
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
            
            # Create an instance of PoseProcessor to use its filtering methods
            from processors.pose_processor import PoseProcessor
            processor = PoseProcessor()
            
            # Apply the selected filter method
            filtered_keypoints = processor.apply_method(keypoints_array, filter_method)
            
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
                # MediaPipe provides 33 keypoints, we need to map to COCO-17
                # MediaPipe keypoint mapping to COCO-17
                mediapipe_to_coco = {
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
                
                coco_keypoints = []
                for coco_idx in range(17):
                    mp_idx = next((k for k, v in mediapipe_to_coco.items() if v == coco_idx), None)
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
                # Sapiens already outputs in COCO format, but we need to ensure dimensions
                # This is a placeholder - adapt based on actual Sapiens output format
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
                
            else:
                print(f"Unsupported landmark source type: {source_type}")
                return [[0, 0, 0, 0] for _ in range(17)]  # Return empty keypoints
                
        except Exception as e:
            print(f"Error extracting COCO keypoints: {str(e)}")
            return [[0, 0, 0, 0] for _ in range(17)]  # Return empty keypoints on error