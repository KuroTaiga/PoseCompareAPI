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

class SapiensProcessor:
    """Processor for Sapiens pose estimation model"""
    
    def __init__(self, 
                 checkpoint=DEFAULT_CHECKPOINT, 
                 device="cuda:0" if torch.cuda.is_available() else "cpu", 
                 batch_size=DEFAULT_BATCH_SIZE, 
                 shape=DEFAULT_SHAPE, 
                 output_folder="static/results", 
                 save_img_flag=True):
        """
        Initialize Sapiens model processor
        
        Args:
            checkpoint: Path to model checkpoint
            device: Device to run model on ('cuda:0' or 'cpu')
            batch_size: Batch size for processing
            shape: Input shape for model (height, width)
            output_folder: Folder to save intermediate results
            save_img_flag: Whether to save intermediate images
        """
        self.save_flag = save_img_flag
        self.checkpoint = checkpoint
        self.device = device
        self.dtype = torch.float32  # TorchScript models use float32
        self.batch_size = batch_size
        self.shape = shape
        self.output_root = output_folder
        
        # Create output directory if needed
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
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
        try:
            # Save original video size and information
            self._save_original_video_size(input_path)
            
            # Resize for processing if needed
            resized_video_path = os.path.splitext(input_path)[0] + "_resized.mp4"
            if not self._resize_video(input_path, resized_video_path):
                return False
                
            # Extract frames from video
            imgs_path = self.vid_to_img(resized_video_path)
            out_img_folder = os.path.join(self.output_root, os.path.basename(imgs_path))
            if not os.path.exists(out_img_folder):
                os.makedirs(out_img_folder)
                
            # Create dataset from image frames
            if os.path.isdir(imgs_path):
                input_dir = imgs_path
                image_names = [
                    img_name for img_name in os.listdir(input_dir) 
                    if img_name.endswith(".jpg") or img_name.endswith(".png")
                ]
                image_names.sort()  # Ensure frames are in order
            else:
                print(f"Error: images path '{imgs_path}' does not exist.")
                return False
                
            # Create input shape from model requirements
            input_shape = (3,)+tuple(self.shape)
            
            # Process frames through the model
            inference_dataset = AdhocImageDataset(
                [os.path.join(input_dir, img_name) for img_name in image_names],
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
            
            # this change was made to accommodate the windows environment
            # window's way of serialize the function is different from linux, which uses fork
            from .model import preprocess_pose_worker, img_save_and_vis_worker
            # Initialize worker pools
            pose_preprocess_pool = WorkerPool(
                preprocess_pose_worker, processes=max(min(self.batch_size, cpu_count()), 1)
            )
            img_save_pool = WorkerPool(
                img_save_and_vis_worker, processes=max(min(self.batch_size, cpu_count()), 1)
            )
            
            # For keypoint tracking
            all_frame_indices = []
            all_coco_keypoints = []
            
            # Process each batch of frames
            for batch_idx, (batch_image_name, batch_orig_imgs, batch_imgs) in enumerate(inference_dataloader):
                # Limit to max_frames
                if len(all_frame_indices) >= max_frames:
                    break
                    
                orig_img_shape = batch_orig_imgs.shape
                valid_images_len = len(batch_orig_imgs)
                bboxes_batch = [[] for _ in range(len(batch_orig_imgs))]
                
                # Create full-frame bounding boxes
                for i, bboxes in enumerate(bboxes_batch):
                    if len(bboxes) == 0:
                        bboxes_batch[i] = np.array(
                            [[0, 0, orig_img_shape[2], orig_img_shape[1]]]  # orig_img_shape: B H W C
                        )
                
                # Map batch indices to number of bboxes
                img_bbox_map = {}
                for i, bboxes in enumerate(bboxes_batch):
                    img_bbox_map[i] = len(bboxes)

                # Preprocess for pose estimation
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
                pose_imgs, pose_img_centers, pose_img_scales = [], [], []
                
                for op in pose_ops:
                    pose_imgs.extend(op[0])
                    pose_img_centers.extend(op[1])
                    pose_img_scales.extend(op[2])

                n_pose_batches = (len(pose_imgs) + self.batch_size - 1) // self.batch_size

                # Run pose estimation on batches
                pose_results = []
                
                for i in range(n_pose_batches):
                    imgs = torch.stack(
                        pose_imgs[i * self.batch_size : (i + 1) * self.batch_size], dim=0
                    )
                    valid_len = len(imgs)
                    imgs = F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, self.batch_size - imgs.shape[0]), value=0)
                    curr_results = self.batch_inference_topdown(self.model, imgs, dtype=self.dtype)[:valid_len]
                    pose_results.extend(curr_results)

                # Group results by image
                batched_results = []
                for _, bbox_len in img_bbox_map.items():
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
                for frame_idx, (img_name, img, results) in enumerate(zip(
                    batch_image_name[:valid_images_len], 
                    batch_orig_imgs[:valid_images_len], 
                    batched_results[:valid_images_len]
                )):
                    # Get actual frame index from filename
                    frame_num = int(os.path.splitext(os.path.basename(img_name))[0])
                    
                    # Only process frames at the sample rate
                    if frame_num % sample_rate == 0:
                        # Extract keypoints from heatmaps
                        keypoints = []
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
                            
                            keypoints.append(coco_kpts)
                        
                        # Add to collection
                        all_frame_indices.append(frame_num)
                        all_coco_keypoints.append(keypoints[0] if keypoints else [])

                # Visualize and save results
                args_list = [
                    (
                        i.numpy(),
                        r,
                        os.path.join(out_img_folder, os.path.basename(img_name)),
                        (input_shape[2], input_shape[1]),
                        4,  # heatmap scale
                        COCO_KPTS_COLORS,
                        kpt_thr,
                        radius,
                        COCO_SKELETON_INFO,
                        thickness,
                    )
                    for i, r, img_name in zip(
                        batch_orig_imgs[:valid_images_len],
                        batched_results[:valid_images_len],
                        batch_image_name,
                    )
                ]
                img_save_pool.run_async(args_list)
            
            # Finish worker pools
            pose_preprocess_pool.finish()
            img_save_pool.finish()
            
            # Apply filtering/interpolation to keypoints if method is not original
            if method != "original" and all_coco_keypoints:
                print(f"Applying {method} to keypoints...")
                
                # Determine if it's a filter or interpolation method
                filter_methods = ['butterworth', 'kalman', 'wiener', 'chebyshev', 'bessel']
                interpolation_methods = ['linear', 'spline', 'bilinear', 'kriging']
                
                if method in filter_methods:
                    # Apply filter
                    all_coco_keypoints = np.array(all_coco_keypoints)
                    all_coco_keypoints = PoseFilter.apply_filter(
                        all_coco_keypoints, 
                        method,
                        filter_window
                    )
                    all_coco_keypoints = all_coco_keypoints.tolist()
                    
                elif method in interpolation_methods:
                    # Apply interpolation
                    all_coco_keypoints = np.array(all_coco_keypoints)
                    all_coco_keypoints = PoseInterpolation.apply_interpolation(
                        all_coco_keypoints,
                        method
                    )
                    all_coco_keypoints = all_coco_keypoints.tolist()
            
            # Generate output video from processed frames
            print("Writing video from processed frames...")
            self.img_to_vid(out_img_folder, output_path)
            
            # Save keypoints to JSON if requested
            if save_keypoints and output_json_path and all_coco_keypoints:
                print(f"Saving {len(all_coco_keypoints)} keypoints frames to {output_json_path}")
                
                # Save keypoints to JSON
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
                
                JointDataProcessor.save_keypoints_to_json(
                    all_coco_keypoints,
                    all_frame_indices,
                    output_json_path,
                    metadata=metadata,
                    sample_rate=sample_rate
                )
            
            # Clean up temporary files if requested
            if not self.save_flag: 
                shutil.rmtree(out_img_folder)
            if os.path.exists(resized_video_path):
                os.remove(resized_video_path)
            
            print(f"Sapiens processing complete. Output saved to {output_path}")
            if save_keypoints and output_json_path:
                print(f"Keypoint data saved to {output_json_path}")
                
            return True
            
        except Exception as e:
            print(f"Error in Sapiens processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def batch_inference_topdown(self, model: nn.Module, imgs: torch.Tensor, dtype=torch.float32, flip=False):
        """Run inference on a batch of images"""
        with torch.no_grad(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype):
            heatmaps = model(imgs.to(self.device))
            if flip:
                heatmaps_ = model(imgs.to(dtype).to(self.device).flip(-1))
                heatmaps = (heatmaps + heatmaps_) * 0.5
            imgs.cpu()
        return heatmaps.cpu()
    
    def _save_original_video_size(self, video_path):
        """
        Saves the original width and height of the input video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.original_width = width
        self.original_height = height
        self.fps = fps
        self.total_frames = total_frames

    def _resize_video(self, input_path, output_path, target_size=None):
        """
        Resizes the input video to the specified size while maintaining aspect ratio.
        """
        if target_size is None:
            target_size = self.shape[::-1]  # Convert (height, width) to (width, height)
            
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return False

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), target_size)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame
            resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            out.write(resized_frame)

        cap.release()
        out.release()
        print(f"Resized video saved to {output_path}")
        return True
    
    def vid_to_img(self, video_path):
        """Extract frames from a video file"""
        if not os.path.exists(video_path):
            print(f"Error: The video file '{video_path}' does not exist.")
            return
        
        # Create output directory
        output_dir = os.path.splitext(video_path)[0]  # Remove extension
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Stop if no more frames
            
            # Construct file name
            filename = os.path.join(output_dir, f"{frame_number:06d}.jpg")
            
            # Save frame as JPEG
            cv2.imwrite(filename, frame)
            frame_number += 1
        
        # Release the video capture object
        cap.release()
        return output_dir
    
    def img_to_vid(self, img_folder, output_path, fps=None):
        """Create a video from a sequence of images"""
        if fps is None:
            fps = self.fps if hasattr(self, 'fps') else 30
            
        images = sorted([f for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')])
        if not images:
            print(f"Error: No images found in '{img_folder}'.")
            return
        
        frame = cv2.imread(os.path.join(img_folder, images[0]))
        height, width, _ = frame.shape
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for image_name in images:
            frame = cv2.imread(os.path.join(img_folder, image_name))
            out.write(frame)
        
        out.release()
        print(f"Video saved to {output_path}")

# def preprocess_pose_worker(orig_img, bboxes_list, input_shape, mean, std):
#     """Preprocess pose images and bboxes."""
#     from .util import top_down_affine_transform
#     import torch
#     import numpy as np
#     import cv2

#     preprocessed_images = []
#     centers = []
#     scales = []
#     for bbox in bboxes_list:
#         img, center, scale = top_down_affine_transform(orig_img.copy(), bbox)
#         img = cv2.resize(img, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
#         img = torch.from_numpy(img)
#         img = img[[2, 1, 0], ...].float()
#         mean = torch.Tensor(mean).view(-1, 1, 1)
#         std = torch.Tensor(std).view(-1, 1, 1)
#         img = (img - mean) / std
#         preprocessed_images.append(img)
#         centers.extend(center)
#         scales.extend(scale)
#     return preprocessed_images, centers, scales
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



def img_save_and_vis_worker(img, results, output_path, input_shape, heatmap_scale, kpt_colors, kpt_thr, radius, skeleton_info, thickness):
    """Save and visualize pose estimation results"""
    import numpy as np
    import torch
    import cv2
    import json
    from .util import udp_decode

    heatmap = results["heatmaps"]
    centres = results["centres"]
    scales = results["scales"]
    instance_keypoints = []
    instance_scores = []

    for i in range(len(heatmap)):
        result = udp_decode(
            heatmap[i].cpu().unsqueeze(0).float().data[0].numpy(),
            input_shape,
            (int(input_shape[0] / heatmap_scale), int(input_shape[1] / heatmap_scale)),
        )

        keypoints, keypoint_scores = result
        keypoints = (keypoints / input_shape) * scales[i] + centres[i] - 0.5 * scales[i]
        instance_keypoints.append(keypoints[0])
        instance_scores.append(keypoint_scores[0])

    pred_save_path = output_path.replace(".jpg", ".json").replace(".png", ".json")
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
            indent="\t",
        )

    instance_keypoints = np.array(instance_keypoints).astype(np.float32)
    instance_scores = np.array(instance_scores).astype(np.float32)

    keypoints_visible = np.ones(instance_keypoints.shape[:-1])
    for kpts, score, visible in zip(instance_keypoints, instance_scores, keypoints_visible):
        if kpt_colors is None or isinstance(kpt_colors, str) or len(kpt_colors) != len(kpts):
            raise ValueError(f"Mismatch in keypoint color length: {len(kpt_colors)} vs {len(kpts)}")

        for kid, kpt in enumerate(kpts):
            if score[kid] < kpt_thr or not visible[kid] or kpt_colors[kid] is None:
                continue
            color = kpt_colors[kid]
            if not isinstance(color, str):
                color = tuple(int(c) for c in color[::-1])
            img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius), color, -1)

        for skid, link_info in skeleton_info.items():
            pt1_idx, pt2_idx = link_info['link']
            color = link_info['color'][::-1]
            pt1 = kpts[pt1_idx]; pt1_score = score[pt1_idx]
            pt2 = kpts[pt2_idx]; pt2_score = score[pt2_idx]
            if pt1_score > kpt_thr and pt2_score > kpt_thr:
                cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, thickness=thickness)

    cv2.imwrite(output_path, img)
