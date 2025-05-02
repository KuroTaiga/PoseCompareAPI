"""
Sapiens pose estimation model processor

This module provides the SapiensProcessor class for processing videos
with the Sapiens pose estimation model, using shared components for
filtering, interpolation, and visualization.
"""

import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import shutil
from multiprocessing import cpu_count
from scipy.signal import butter, filtfilt

# Import Sapiens-specific constants and utilities
from .classes_and_consts import (
    COCO_KPTS_COLORS, 
    COCO_SKELETON_INFO,
    DEFAULT_CHECKPOINT,
    DEFAULT_SHAPE,
    DEFAULT_MEAN,
    DEFAULT_STD,
    DEFAULT_BATCH_SIZE,
    DEFAULT_FILTER_WINDOW,
    DEFAULT_KPT_THRESHOLD
)
from .util import udp_decode, top_down_affine_transform

# Import shared components
from utils.joint_data import JointDataProcessor
from processors.noise_filters import PoseFilter
from processors.interpolation import PoseInterpolation
from utils.visualization import PoseVisualizer
from .worker_pool import WorkerPool

class AdhocImageDataset(torch.utils.data.Dataset):
    """Dataset for processing images without predefined transformations"""
    def __init__(self, image_list, shape=None, mean=None, std=None):
        self.image_list = image_list
        if shape:
            assert len(shape) == 2
        if mean or std:
            assert len(mean) == 3
            assert len(std) == 3
        self.shape = shape
        self.mean = torch.tensor(mean) if mean else None
        self.std = torch.tensor(std) if std else None

    def __len__(self):
        return len(self.image_list)
    
    def _preprocess(self, img):
        if self.shape:
            img = cv2.resize(img, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_LINEAR)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img[[2, 1, 0], ...].float()
        if self.mean is not None and self.std is not None:
            mean=self.mean.view(-1, 1, 1)
            std=self.std.view(-1, 1, 1)
            img = (img - mean) / std
        return img
    
    def __getitem__(self, idx):
        orig_img_dir = self.image_list[idx]
        orig_img = cv2.imread(orig_img_dir)
        img = self._preprocess(orig_img)
        return orig_img_dir, orig_img, img

def scale_keypoints_to_original(keypoints, original_width, original_height, resized_width, resized_height):
    """
    Scale keypoints from resized video dimensions to original video dimensions
    
    Args:
        keypoints: List of keypoints [x, y, z, confidence]
        original_width: Original video width
        original_height: Original video height
        resized_width: Resized video width
        resized_height: Resized video height
        
    Returns:
        Scaled keypoints
    """
    scaled_keypoints = []
    
    for kp in keypoints:
        # Extract coordinates and confidence
        x, y, z, conf = kp
        
        # Scale x and y to original dimensions
        x_scaled = (x / resized_width) * original_width
        y_scaled = (y / resized_height) * original_height
        
        # Add scaled keypoint
        scaled_keypoints.append([x_scaled, y_scaled, z, conf])
    
    return scaled_keypoints

class SapiensProcessor:
    """Processor for Sapiens pose estimation model"""
    
    _model_cache = {}  # Cache for loaded models to avoid reloading
    def __init__(self, 
                 checkpoint=DEFAULT_CHECKPOINT, 
                 device="cuda:0" if torch.cuda.is_available() else "cpu", 
                 batch_size=DEFAULT_BATCH_SIZE, 
                 shape=DEFAULT_SHAPE, 
                 output_folder="static/results", 
                 save_img_flag=True,
                 session_id=None,
                 upload_id=None,
                 model_id=None,
                 job_id = None,
                 filter_method="original"):
        """
        Initialize Sapiens model processor
        
        Args:
            checkpoint: Path to model checkpoint
            device: Device to run model on ('cuda:0' or 'cpu')
            batch_size: Batch size for processing
            shape: Input shape for model (height, width)
            output_folder: Base folder to save intermediate results
            save_img_flag: Whether to save intermediate images
            session_id: Current session ID (for managing temp files)
            upload_id: Current upload ID (for managing temp files)
            model_id: Model identifier (for managing temp files)
            filter_method: Filtering or interpolation method (for managing temp files)
        """
        
        self.save_flag = save_img_flag
        self.checkpoint = checkpoint
        self.device = device
        self.dtype = torch.float32  # TorchScript models use float32
        self.batch_size = batch_size
        self.shape = shape
        self.filter_method = filter_method
        
        # Create job-specific path components
        self.session_id = session_id
        self.upload_id = upload_id
        self.model_id = model_id or "sapiens"
        self.job_id = job_id
        
        # Build the output root path
        self._build_output_path(output_folder)
        
        cache_key = f"{checkpoint}_{device}"
        if cache_key in SapiensProcessor._model_cache:
            print(f"Using cached Sapiens model for {checkpoint}")
            self.model = SapiensProcessor._model_cache[cache_key]
        else:
            # Load the model
            try:
                print(f"Loading Sapiens model from {checkpoint}")
                self.model = torch.jit.load(checkpoint)
                self.model = self.model.to(self.device)
                self.model.eval()
                print(f"Sapiens model loaded successfully")
            except Exception as e:
                print(f"Failed to load Sapiens model: {e}")
                raise
            
    def _build_output_path(self, base_folder):
        """Build the output path using session, upload, model and filter information"""
        if self.session_id and self.upload_id:
            # Use structure like: base_folder/session_id/upload_id/model_id_filter_method/
            filter_suffix = f"_{self.filter_method}" if self.filter_method != "original" else ""
            # job_id = f"{self.model_id}{filter_suffix}"
            
            self.output_root = os.path.join(base_folder, str(self.session_id), str(self.upload_id), str(self.job_id))
        else:
            self.output_root = base_folder
            
        # Create the directory if it doesn't exist
        os.makedirs(self.output_root, exist_ok=True)
        print(f"Temporary files will be saved to: {self.output_root}")

    def process_video(self, 
                      input_path, 
                      output_path, 
                      method="original", 
                      filter_window=DEFAULT_FILTER_WINDOW, 
                      output_json_path=None,
                      output_format="mp4", 
                      kpt_thr=DEFAULT_KPT_THRESHOLD, 
                      radius=6, 
                      thickness=3,
                      save_keypoints=True, 
                      sample_rate=1,
                      max_frames=1000):
        """
        Process a video with the Sapiens model
        
        Args:
            input_path: Path to input video
            output_path: Path to save processed video
            method: Filtering or interpolation method ('original', 'butterworth', etc.)
            filter_window: Window size for filtering
            output_json_path: Path to save keypoint data (if save_keypoints is True)
            output_format: Output video format
            kpt_thr: Confidence threshold for keypoints
            radius: Radius of keypoint circles
            thickness: Line thickness for skeleton
            save_keypoints: Whether to save keypoint data as JSON
            sample_rate: Save every Nth frame's keypoints
            max_frames: Maximum number of frames to process
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Update filter method if different from initialization
        if method != self.filter_method:
            self.filter_method = method
            # Rebuild output path with new filter method
            base_folder = os.path.dirname(os.path.dirname(os.path.dirname(self.output_root)))
            self._build_output_path(base_folder)
            
        try:
            # Create job-specific temporary paths
            video_basename = os.path.basename(os.path.splitext(input_path)[0])
            job_temp_dir = os.path.join(self.output_root,"temp", video_basename)
            os.makedirs(job_temp_dir, exist_ok=True)
            
            resized_video_path = os.path.join(job_temp_dir, f"{video_basename}_resized.mp4")
            frames_dir = os.path.join(job_temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Make sure output directories exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if output_json_path:
                os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            
            # Step 1: Save original video size and information
            self._save_original_video_size(input_path)
            
            # Step 2: Resize video for processing
            print(f"Resizing video to {self.shape[1]}x{self.shape[0]}...")
            if not self._resize_video(input_path, resized_video_path):
                return False
                
            # Step 3: Extract frames from video
            print(f"Extracting frames from video...")
            self._extract_frames(resized_video_path, frames_dir)
                
            # Step 4: Create dataset from frames
            frame_paths = sorted([
                os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
                if f.endswith('.jpg') or f.endswith('.png')
            ])
            
            if not frame_paths:
                print(f"Error: No frames were extracted from the video.")
                return False
                
            # Step 5: Process frames through the model
            print(f"Processing {len(frame_paths)} frames with Sapiens model...")
            all_frame_indices, all_keypoints = self._process_frames(
                frame_paths, 
                max_frames=max_frames, 
                sample_rate=sample_rate,
                kpt_thr=kpt_thr,
                radius=radius,
                thickness=thickness
            )
            
            # Step 6: Apply filtering/interpolation if needed
            if method != "original" and all_keypoints:
                all_keypoints = self._apply_filter_or_interpolation(
                    all_keypoints,
                    method,
                    filter_window
                )
            
            # Step 7: Generate output video
            print(f"Generating output video...")
            processed_frames_dir = os.path.join(job_temp_dir, "processed_frames")
            self._create_output_video(processed_frames_dir, output_path)
            
            # Step 8: Save keypoints to JSON if requested
            if save_keypoints and output_json_path and all_keypoints:
                self._save_keypoints_json(
                    all_keypoints,
                    all_frame_indices,
                    output_json_path,
                    input_path,
                    method,
                    filter_window,
                    sample_rate
                )
            
            # Step 9: Clean up temporary files if requested
            if not self.save_flag:
                self._cleanup_temp_files(job_temp_dir)
            
            print(f"Sapiens processing complete. Output saved to {output_path}")
            if save_keypoints and output_json_path:
                print(f"Keypoint data saved to {output_json_path}")
                
            return True
            
        except Exception as e:
            print(f"Error in Sapiens processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _save_original_video_size(self, video_path):
        """Save the original width and height of the input video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        self.original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print(f"Original video dimensions: {self.original_width}x{self.original_height}, {self.fps} FPS, {self.total_frames} frames")
        return True

    def _resize_video(self, input_path, output_path):
        """Resize the input video to the model's input dimensions"""
        target_size = self.shape[::-1]  # Convert (height, width) to (width, height)
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return False
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Configure video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), target_size)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame
            resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            out.write(resized_frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        print(f"Resized video saved to {output_path} ({frame_count} frames)")
        return True

    def _extract_frames(self, video_path, output_dir):
        """Extract frames from a video file"""
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' does not exist.")
            return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame as JPEG
            cv2.imwrite(os.path.join(output_dir, f"{frame_number:06d}.jpg"), frame)
            frame_number += 1
        
        cap.release()
        print(f"Extracted {frame_number} frames to {output_dir}")
        return True

    def _process_frames(self, frame_paths, max_frames=1000, sample_rate=1, kpt_thr=0.3, radius=4, thickness=2):
        """Process frames through the Sapiens model to extract keypoints"""
        # Create processing dataset
        inference_dataset = AdhocImageDataset(
            frame_paths[:max_frames],
            shape=self.shape,
            mean=DEFAULT_MEAN,
            std=DEFAULT_STD
        )
        
        inference_dataloader = torch.utils.data.DataLoader(
            inference_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=max(min(self.batch_size, cpu_count()) // 4, 4),
        )
        
        # Initialize worker pools for parallel processing
        pose_preprocess_pool = WorkerPool(
            preprocess_pose_worker, processes=max(min(self.batch_size, cpu_count()), 1)
        )
        img_save_pool = WorkerPool(
            img_save_and_vis_worker, processes=max(min(self.batch_size, cpu_count()), 1)
        )
        
        # Initialize storage for frame indices and keypoints
        all_frame_indices = []
        all_coco_keypoints = []
        
        # Process each batch of frames
        input_shape = (3,) + tuple(self.shape)
        
        for batch_idx, (batch_image_name, batch_orig_imgs, batch_imgs) in enumerate(inference_dataloader):
            # Skip if we've reached max frames
            if len(all_frame_indices) >= max_frames:
                break
                
            # Prepare data for processing
            valid_images_len = len(batch_orig_imgs)
            bboxes_batch = [
                np.array([[0, 0, batch_orig_imgs.shape[2], batch_orig_imgs.shape[1]]])
                for _ in range(valid_images_len)
            ]
            
            # Map batch indices to number of bboxes
            img_bbox_map = {i: 1 for i in range(valid_images_len)}

            # Preprocess frames for pose estimation
            args_list = [
                (
                    i,
                    bbox_list,
                    (input_shape[1], input_shape[2]),
                    DEFAULT_MEAN,
                    DEFAULT_STD,
                )
                for i, bbox_list in zip(batch_orig_imgs.numpy(), bboxes_batch)
            ]
            pose_ops = pose_preprocess_pool.run(args_list)
            
            # Collect preprocessing results
            pose_imgs, pose_img_centers, pose_img_scales = [], [], []
            for op in pose_ops:
                pose_imgs.extend(op[0])
                pose_img_centers.extend(op[1])
                pose_img_scales.extend(op[2])

            # Process pose images in batches
            n_pose_batches = (len(pose_imgs) + self.batch_size - 1) // self.batch_size
            pose_results = []
            
            for i in range(n_pose_batches):
                batch_start = i * self.batch_size
                batch_end = min((i + 1) * self.batch_size, len(pose_imgs))
                valid_len = batch_end - batch_start
                
                # Stack images for batch processing
                imgs = torch.stack(pose_imgs[batch_start:batch_end], dim=0)
                
                # Pad batch if needed
                if valid_len < self.batch_size:
                    imgs = F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, self.batch_size - valid_len), value=0)
                
                # Run inference
                curr_results = self.batch_inference_topdown(self.model, imgs, dtype=self.dtype)[:valid_len]
                pose_results.extend(curr_results)

            # Group results by image
            batched_results = []
            for img_idx, bbox_len in img_bbox_map.items():
                result = {
                    "heatmaps": pose_results[:bbox_len].copy(),
                    "centres": pose_img_centers[:bbox_len].copy(),
                    "scales": pose_img_scales[:bbox_len].copy(),
                }
                batched_results.append(result)
                del (
                    pose_results[:bbox_len],
                    pose_img_centers[:bbox_len],
                    pose_img_scales[:bbox_len],
                )

            # Extract keypoints for each frame
            for frame_idx, (img_path, img, results) in enumerate(zip(
                batch_image_name[:valid_images_len], 
                batch_orig_imgs[:valid_images_len], 
                batched_results[:valid_images_len]
            )):
                # Get frame number from filename
                frame_num = int(os.path.splitext(os.path.basename(img_path))[0])
                
                # Only process frames at the sample rate
                if frame_num % sample_rate == 0:
                    # Extract keypoints from heatmaps
                    for heatmap, centre, scale in zip(
                        results["heatmaps"], 
                        results["centres"],
                        results["scales"]
                    ):
                        # Decode keypoints from heatmap
                        kpts, scores = udp_decode(
                            heatmap.cpu().unsqueeze(0).float().data[0].numpy(),
                            input_shape[1:],
                            (int(input_shape[1] / 4), int(input_shape[2] / 4)),
                        )
                        
                        # Convert keypoints to image coordinates
                        kpts = (kpts / input_shape[1:]) * scale + centre - 0.5 * scale
                        
                        # Format as COCO keypoints with confidence
                        coco_kpts = []
                        for i, (kpt, score) in enumerate(zip(kpts[0], scores[0])):
                            coco_kpts.append([float(kpt[0]), float(kpt[1]), 0.0, float(score)])
                        
                        # Add to collections
                        all_frame_indices.append(frame_num)
                        all_coco_keypoints.append(coco_kpts)
                        
                        # Visualize and save processed frame
                        processed_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), "processed_frames")
                        os.makedirs(processed_dir, exist_ok=True)
                        
                        # Prepare arguments for visualization
                        args = (
                            img.numpy(),
                            results,
                            os.path.join(processed_dir, os.path.basename(img_path)),
                            (input_shape[2], input_shape[1]),
                            4,  # heatmap scale
                            COCO_KPTS_COLORS,
                            kpt_thr,
                            radius,
                            COCO_SKELETON_INFO,
                            thickness,
                            self.original_width,
                            self.original_height,
                        )
                        
                        # Save visualization asynchronously
                        img_save_pool.run_async([args])
        
        # Wait for all workers to finish
        pose_preprocess_pool.finish()
        img_save_pool.finish()
        
        print(f"Processed {len(all_frame_indices)} frames and extracted keypoints")
        return all_frame_indices, all_coco_keypoints

    def _apply_filter_or_interpolation(self, keypoints, method, filter_window):
        """Apply filtering or interpolation to keypoints"""
        print(f"Applying {method} to keypoints...")
        
        # Convert to numpy array for processing
        keypoints_array = np.array(keypoints)
        
        # Determine if it's a filter or interpolation method
        filter_methods = ['butterworth', 'kalman', 'wiener', 'chebyshev', 'bessel']
        interpolation_methods = ['linear', 'spline', 'bilinear', 'kriging']
        
        if method in filter_methods:
            # Apply filter
            filtered_keypoints = PoseFilter.apply_filter(
                keypoints_array, 
                method,
                filter_window
            )
            return filtered_keypoints.tolist()
            
        elif method in interpolation_methods:
            # Apply interpolation
            interpolated_keypoints = PoseInterpolation.apply_interpolation(
                keypoints_array,
                method
            )
            return interpolated_keypoints.tolist()
            
        # If method is not recognized, return original keypoints
        return keypoints

    def _create_output_video(self, frames_dir, output_path):
        """Create output video from processed frames"""
        if not os.path.exists(frames_dir):
            print(f"Warning: Processed frames directory not found, trying to create output video from original frames")
            frames_dir = os.path.join(os.path.dirname(frames_dir), "frames")
            if not os.path.exists(frames_dir):
                print(f"Error: Could not find frames to create output video")
                return False
        
        # Get list of frame images
        images = sorted([
            f for f in os.listdir(frames_dir) 
            if f.endswith('.jpg') or f.endswith('.png')
        ])
        
        if not images:
            print(f"Error: No images found in '{frames_dir}'")
            return False
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(os.path.join(frames_dir, images[0]))
        height, width, _ = first_frame.shape
        
        # Set target dimensions to original video dimensions
        target_size = (self.original_width, self.original_height)
        needs_resize = (width != self.original_width) or (height != self.original_height)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Configure video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, target_size)
        
        # Process each frame
        for image_name in images:
            frame = cv2.imread(os.path.join(frames_dir, image_name))
            
            # Resize if needed
            if needs_resize:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
                
            out.write(frame)
        
        out.release()
        print(f"Output video created: {output_path}")
        return True

    def _save_keypoints_json(self, keypoints, frame_indices, output_path, input_path, method, filter_window, sample_rate):
        """Save keypoints to JSON file"""
        print(f"Saving keypoints to {output_path}")
        
        # Scale keypoints to original video dimensions
        scaled_keypoints = []
        for frame_keypoints in keypoints:
            scaled_frame_keypoints = scale_keypoints_to_original(
                frame_keypoints,
                self.original_width,
                self.original_height,
                self.shape[1],  # resized width
                self.shape[0]   # resized height
            )
            scaled_keypoints.append(scaled_frame_keypoints)
        
        # Create metadata
        metadata = {
            "model": os.path.basename(self.checkpoint),
            "method": method,
            "filter_window": filter_window,
            "video_path": input_path,
            "frame_width": self.original_width,
            "frame_height": self.original_height,
            "fps": self.fps,
            "total_frames": self.total_frames
        }
        
        # Save to JSON
        JointDataProcessor.save_keypoints_to_json(
            scaled_keypoints,
            frame_indices,
            output_path,
            metadata=metadata,
            sample_rate=sample_rate
        )
        
        return True

    def _cleanup_temp_files(self, temp_dir):
        """Clean up temporary files and directories"""
        print(f"Cleaning up temporary files in {temp_dir}")
        
        # Remove the temporary directory and all its contents
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")
        
        # Try to remove parent directories if they're empty
        try:
            parent_dir = os.path.dirname(temp_dir)
            if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                os.rmdir(parent_dir)
                print(f"Removed empty directory: {parent_dir}")
            
            # Check one level up
            grandparent_dir = os.path.dirname(parent_dir)
            if os.path.exists(grandparent_dir) and not os.listdir(grandparent_dir):
                os.rmdir(grandparent_dir)
                print(f"Removed empty directory: {grandparent_dir}")
        except Exception as e:
            print(f"Note: Could not remove parent directories: {str(e)}")
        
        print("Temporary files cleanup complete")
        return True

    def batch_inference_topdown(self, model: nn.Module, imgs: torch.Tensor, dtype=torch.float32, flip=False):
        """Run inference on a batch of images"""
        with torch.no_grad(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype):
            # Move images to the correct device
            imgs_device = imgs.to(self.device)
            
            # Run the model
            heatmaps = model(imgs_device)
            
            # Add flip augmentation if requested
            if flip:
                heatmaps_ = model(imgs_device.flip(-1))
                heatmaps = (heatmaps + heatmaps_) * 0.5
            
            # Free GPU memory
            imgs_device = None
            
            # Return heatmaps on CPU
            return heatmaps.cpu()

# Worker functions (kept outside the class for multiprocessing compatibility)
def preprocess_pose_worker(orig_img, bboxes_list, input_shape, mean, std):
    """Preprocess pose images using fixed center and scale."""
    import torch
    import numpy as np
    import cv2

    preprocessed_images = []
    centers = []
    scales = []
    
    # Force center and scale to be consistent for all frames
    H, W, _ = orig_img.shape
    center = np.array([W / 2, H / 2])
    scale = max(H, W) * 1.0  # scale matches full frame
    
    for _ in bboxes_list:
        # Crop center with identity transform (no actual cropping)
        img = cv2.resize(orig_img.copy(), (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img[[2, 1, 0], ...].float()
        mean = torch.Tensor(mean).view(-1, 1, 1)
        std = torch.Tensor(std).view(-1, 1, 1)
        img = (img - mean) / std
        preprocessed_images.append(img)
        centers.append(center)
        scales.append(scale)
    return preprocessed_images, centers, scales

def img_save_and_vis_worker(img, results, output_path, input_shape, heatmap_scale, kpt_colors, kpt_thr, radius, skeleton_info, thickness, out_width, out_height):
    """Save and visualize pose estimation results"""
    import numpy as np
    import cv2
    import json
    import os
    from .util import udp_decode
    
    # Ensure output directory exists
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except Exception as e:
        print(f"Warning: Failed to create output directory for {output_path}: {e}")
        # Try to create parent directories one by one
        path_parts = os.path.dirname(output_path).split(os.path.sep)
        current_path = ""
        for part in path_parts:
            if not part:  # Skip empty parts (like after a leading slash)
                continue
            if current_path:
                current_path = os.path.join(current_path, part)
            else:
                current_path = part
            try:
                if not os.path.exists(current_path):
                    os.mkdir(current_path)
            except Exception as nested_e:
                print(f"Warning: Could not create directory {current_path}: {nested_e}")

    heatmap = results["heatmaps"]
    instance_keypoints = []
    instance_scores = []

    # Get the dimensions of the output image for proper scaling
    H, W = img.shape[:2]  # Height and width of the current frame
    
    # We'll resize the current frame to match the output dimensions
    if out_width != W or out_height != H:
        # Only resize once and do all drawing on the properly sized image
        img = cv2.resize(img, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
    
    for i in range(len(heatmap)):
        result = udp_decode(
            heatmap[i].cpu().unsqueeze(0).float().data[0].numpy(),
            input_shape,
            (int(input_shape[0] / heatmap_scale), int(input_shape[1] / heatmap_scale)),
        )

        keypoints, keypoint_scores = result
        
        # Apply a consistent direct scaling approach
        # Scale directly from model space to output dimensions
        keypoints = keypoints.copy()  # Make a copy to avoid modifying the original
        keypoints[0, :, 0] = (keypoints[0, :, 0] / input_shape[0]) * out_width
        keypoints[0, :, 1] = (keypoints[0, :, 1] / input_shape[1]) * out_height
        
        # Store the keypoints for JSON output
        instance_keypoints.append(keypoints[0])
        instance_scores.append(keypoint_scores[0])

    # Save keypoint data to JSON
    pred_save_path = output_path.replace(".jpg", ".json").replace(".png", ".json")
    os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
    
    with open(pred_save_path, "w") as f:
        json.dump(
            dict(
                instance_info=[
                    {
                        "keypoints": keypoints.tolist(),
                        "keypoint_scores": keypoint_scores.tolist(),
                    }
                    for keypoints, keypoint_scores in zip(instance_keypoints, instance_scores)
                ]
            ),
            f,
            indent=2
        )

    # Convert to numpy arrays for visualization
    instance_keypoints = np.array(instance_keypoints).astype(np.float32)
    instance_scores = np.array(instance_scores).astype(np.float32)
    
    # Draw keypoints and skeleton
    keypoints_visible = np.ones(instance_keypoints.shape[:-1])
    for kpts, score, visible in zip(instance_keypoints, instance_scores, keypoints_visible):
        # Validate keypoint colors
        if kpt_colors is None or isinstance(kpt_colors, str) or len(kpt_colors) != len(kpts):
            raise ValueError(f"Mismatch in keypoint color length: {len(kpt_colors)} vs {len(kpts)}")

        # Draw keypoints
        for kid, kpt in enumerate(kpts):
            if score[kid] < kpt_thr or not visible[kid] or kpt_colors[kid] is None:
                continue
            color = kpt_colors[kid]
            if not isinstance(color, str):
                color = tuple(int(c) for c in color[::-1])  # Convert RGB to BGR
            
            # Draw circle at keypoint location
            x, y = int(kpt[0]), int(kpt[1])
            cv2.circle(img, (x, y), int(radius), color, -1)

        # Draw skeleton connections
        for skid, link_info in skeleton_info.items():
            pt1_idx, pt2_idx = link_info['link']
            color = link_info['color'][::-1]  # Convert RGB to BGR
            
            pt1 = kpts[pt1_idx]; pt1_score = score[pt1_idx]
            pt2 = kpts[pt2_idx]; pt2_score = score[pt2_idx]
            
            # Only draw line if both keypoints are confident enough
            if pt1_score > kpt_thr and pt2_score > kpt_thr:
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])
                cv2.line(img, (x1, y1), (x2, y2), color, thickness=thickness)

    # Save the processed image
    cv2.imwrite(output_path, img)