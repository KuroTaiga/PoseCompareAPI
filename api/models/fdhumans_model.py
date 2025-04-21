import cv2
import torch
import numpy as np
from scipy import signal
import logging
import os
from FDHumans.hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from FDHumans.hmr2.utils import recursive_to
from FDHumans.hmr2.utils.renderer import Renderer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FourDHumanWrapper:
    """Wrapper for 4DHuman model."""
    def __init__(self, checkpoint_path=None, device=None):
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
        try:
            frame_resized = cv2.resize(frame, (256, 256))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) / 255.0
            input_tensor = torch.tensor(frame_rgb.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
            return input_tensor.to(self.device)
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}", exc_info=True)
            return None

    def process_frame(self, frame, show_background=True, noise_filter='None'):
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
                    if noise_filter == 'butterworth':
                        display_frame = self._apply_butterworth(display_frame)
                    elif noise_filter == 'chebyshev':
                        display_frame = self._apply_chebyshev(display_frame)
                    elif noise_filter == 'bessel':
                        display_frame = self._apply_bessel(display_frame)
                        
            return display_frame
        except Exception as e:
            logger.error(f"Error processing frame with 4DHuman model: {e}", exc_info=True)
            return frame

    def _apply_chebyshev(self, frame, order=4, ripple_db=1.0, cutoff=0.1):
        try:
            b, a = signal.cheby1(order, ripple_db, cutoff, 'low')
            smoothed_frame = np.zeros_like(frame, dtype=np.float32)
            for i in range(frame.shape[2]):  # Iterate over RGB channels
                smoothed_frame[:, :, i] = signal.filtfilt(b, a, frame[:, :, i].astype(np.float32), axis=0)
            # Clip values back to valid range for image data
            smoothed_frame = np.clip(smoothed_frame, 0, 255).astype(np.uint8)
            return smoothed_frame
        except Exception as e:
            logger.error(f"Error applying chebyshev filter: {e}")
            return frame

    def _apply_bessel(self, frame, order=4, cutoff=0.1):
        try:
            b, a = signal.bessel(order, cutoff, 'low')
            smoothed_frame = np.zeros_like(frame, dtype=np.float32)
            for i in range(frame.shape[2]):  # Iterate over RGB channels
                smoothed_frame[:, :, i] = signal.filtfilt(b, a, frame[:, :, i].astype(np.float32), axis=0)
            # Clip values back to valid range for image data
            smoothed_frame = np.clip(smoothed_frame, 0, 255).astype(np.uint8)
            return smoothed_frame
        except Exception as e:
            logger.error(f"Error applying bessel filter: {e}")
            return frame

    def _apply_butterworth(self, frame, order=4, cutoff=0.1):
        try:
            b, a = signal.butter(order, cutoff, 'low')
            smoothed_frame = np.zeros_like(frame, dtype=np.float32)
            for i in range(frame.shape[2]):  # Iterate over RGB channels
                smoothed_frame[:, :, i] = signal.filtfilt(b, a, frame[:, :, i].astype(np.float32), axis=0)
            # Clip values back to valid range for image data
            smoothed_frame = np.clip(smoothed_frame, 0, 255).astype(np.uint8)
            return smoothed_frame
        except Exception as e:
            logger.error(f"Error applying butterworth filter: {e}")
            return frame

    def render_mesh(self, frame, vertices, camera_params):
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

    def process_video(self, video_path, output_path, method='original', show_background=True):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_path}")
                
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            frame_count = 0 
            max_frames = 300  # Limit processing to 300 frames

            # Process frames
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                logger.info(f"Processing frame {frame_count}")

                processed_frame = self.process_frame(frame, show_background, method)
                video_writer.write(processed_frame)
                
            cap.release()
            video_writer.release()
            
            logger.info(f"Video processing complete. Output saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing video: {e}", exc_info=True)
            return False