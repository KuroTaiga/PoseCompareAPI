"""
4DHumans pose estimation model processor

This module provides the FourDHumanWrapper class for processing videos
with the 4DHumans pose estimation model, using shared components for
filtering and visualization.
"""

import cv2
import torch
import numpy as np
import logging
import os
from FDHumans.hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from FDHumans.hmr2.utils import recursive_to
from FDHumans.hmr2.utils.renderer import Renderer
from processors.noise_filters import PoseFilter
from utils.joint_data import JointDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FourDHumanWrapper:
    """Wrapper for 4DHuman model."""
    def __init__(self, checkpoint_path=None, device=None):
        """
        Initialize 4DHuman model
        
        Args:
            checkpoint_path: Path to model checkpoint (defaults to DEFAULT_CHECKPOINT)
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT

        try:
            self.model, self.model_cfg = load_hmr2(self.checkpoint_path)
            self.model.to(self.device)
            self.model.eval()
            self.render = Renderer(self.model_cfg, faces=self.model.smpl.faces)
            logger.info("4DHuman model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize 4DHuman model: {e}", exc_info=True)
            raise

    def preprocess_frame(self, frame):
        """
        Preprocess a single frame for the model
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame tensor
        """
        try:
            frame_resized = cv2.resize(frame, (256, 256))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) / 255.0
            input_tensor = torch.tensor(frame_rgb.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
            return input_tensor.to(self.device)
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}", exc_info=True)
            return None

    def process_frame(self, frame, show_background=True, noise_filter='original'):
        """
        Process a single frame with the 4DHuman model
        
        Args:
            frame: Input frame
            show_background: Whether to show the background in the output
            noise_filter: Filter to apply ('original', 'butterworth', etc.)
            
        Returns:
            Processed frame with mesh overlay
        """
        try:
            display_frame = frame.copy() if show_background else np.zeros(frame.shape, dtype=np.uint8)
            
            input_tensor = self.preprocess_frame(frame)
            
            if input_tensor is not None:
                with torch.no_grad():
                    batch = {"img": input_tensor}
                    batch = recursive_to(batch, self.device)
                    output = self.model(batch)
                    
                    vertices = output["pred_vertices"][0].cpu().numpy()
                    camera_params = output["pred_cam"][0].cpu().numpy()
                    
                    display_frame = self.render_mesh(display_frame, vertices, camera_params)
                    
                    # Apply noise filter if specified
                    if noise_filter != 'original':
                        # Convert the frame to a format suitable for filtering
                        frame_data = np.array([[display_frame]])
                        
                        # Apply the filter
                        filtered_data = PoseFilter.apply_filter(frame_data, noise_filter)
                        
                        # Extract the filtered frame
                        display_frame = filtered_data[0, 0]
                        
                        # Ensure the frame is in the correct format
                        display_frame = np.clip(display_frame, 0, 255).astype(np.uint8)
                        
            return display_frame
        except Exception as e:
            logger.error(f"Error processing frame with 4DHuman model: {e}", exc_info=True)
            return frame

    def render_mesh(self, frame, vertices, camera_params):
        """
        Render 3D mesh on a frame
        
        Args:
            frame: Input frame
            vertices: 3D vertices from model output
            camera_params: Camera parameters
            
        Returns:
            Frame with mesh overlay
        """
        try:
            s, tx, ty = camera_params
            img_h, img_w = frame.shape[:2]
            black_frame = np.zeros(frame.shape, dtype=np.uint8)
            
            projected_vertices = vertices[:, :2] * s + np.array([tx, ty])
            projected_vertices[:, 0] = (projected_vertices[:, 0] + 1) * img_w / 2.0
            projected_vertices[:, 1] = img_h-(1-projected_vertices[:, 1]) * img_h / 2.0

            for v in projected_vertices.astype(int):
                cv2.circle(black_frame, tuple(v), 2, (0, 255, 0), -1)
            
            return black_frame
        except Exception as e:
            logger.error(f"Error rendering mesh: {e}", exc_info=True)
            return frame

    def process_video(self, input_path, output_path, method='original', 
                      filter_window=5, output_json_path=None,
                      show_background=True, max_frames=300):
        """
        Process a video with the 4DHuman model
        
        Args:
            input_path: Path to input video
            output_path: Path to save output video
            method: Filter method to apply
            filter_window: Window size for filter (if applicable)
            output_json_path: Path to save keypoint data (optional)
            show_background: Whether to show the background
            max_frames: Maximum number of frames to process
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {input_path}")
                
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            frame_count = 0
            max_frames = min(max_frames, total_frames)
            
            # For keypoint extraction (if JSON output is requested)
            all_keypoints = []
            frame_indices = []

            # Process frames
            logger.info(f"Processing video with 4DHuman model ({method})")
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                logger.info(f"Processing frame {frame_count}/{max_frames}")

                # Process the frame
                processed_frame = self.process_frame(frame, show_background, method)
                video_writer.write(processed_frame)
                
                # Extract keypoints for JSON output if requested
                if output_json_path:
                    # This is a placeholder - actual keypoint extraction would depend
                    # on how 4DHuman represents keypoints
                    # For now, we'll just add empty keypoints
                    all_keypoints.append([[0, 0, 0, 0] for _ in range(17)])  # COCO format
                    frame_indices.append(frame_count)
            
            cap.release()
            video_writer.release()
            
            # Save keypoints to JSON if requested
            if output_json_path and all_keypoints:
                metadata = {
                    "model": "4dhumans",
                    "method": method,
                    "filter_window": filter_window,
                    "video_path": input_path,
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                    "fps": fps,
                    "total_frames": total_frames
                }
                
                JointDataProcessor.save_keypoints_to_json(
                    all_keypoints,
                    frame_indices,
                    output_json_path,
                    metadata=metadata
                )
            
            logger.info(f"Video processing complete. Output saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing video: {e}", exc_info=True)
            return False