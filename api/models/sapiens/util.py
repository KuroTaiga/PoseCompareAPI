"""
Utility functions for the Sapiens pose estimation model

This module contains helper functions used by the Sapiens model processor,
including keypoint decoding and transformation utilities.
"""

import numpy as np
import cv2

def udp_decode(heatmap, input_shape, hidden_dim):
    """
    Decode keypoints from heatmaps using UDP (Unbiased Data Processing)
    
    Args:
        heatmap: Heatmap tensor [channels, height, width]
        input_shape: Shape of the input image (height, width)
        hidden_dim: Hidden dimension size (height, width)
        
    Returns:
        Tuple of keypoints and scores (keypoints, scores)
    """
    keypoints = []
    scores = []
    
    # Extract heatmap dimensions
    c, h, w = heatmap.shape
    
    # Create mesh grid for coordinate computation
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    
    for i in range(c):
        # Get heatmap for current keypoint
        hm = heatmap[i]
        
        # Get score (max value in heatmap)
        score = np.max(hm)
        scores.append(score)
        
        if score > 0:
            # Calculate weighted average for sub-pixel precision
            xy_coords = np.array([
                np.sum(xx * hm) / np.sum(hm),
                np.sum(yy * hm) / np.sum(hm)
            ])
            
            # Convert to original image space
            xy_coords[0] = xy_coords[0] * input_shape[1] / hidden_dim[1]
            xy_coords[1] = xy_coords[1] * input_shape[0] / hidden_dim[0]
            
            keypoints.append(xy_coords)
        else:
            # If no detection, use center of heatmap
            keypoints.append(np.array([input_shape[1] / 2, input_shape[0] / 2]))
    
    # Reshape to expected format
    keypoints = np.array(keypoints)[np.newaxis, ...]
    scores = np.array(scores)[np.newaxis, ...]
    
    return keypoints, scores

def top_down_affine_transform(img, bbox):
    """
    Apply affine transformation for top-down pose estimation
    
    Args:
        img: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Tuple of (transformed_image, center, scale)
    """
    # Calculate center and scale from bbox
    x1, y1, x2, y2 = bbox
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    scale = np.array([(x2 - x1), (y2 - y1)])
    
    # Adjust scale and center for square crop
    if scale[0] > scale[1]:
        scale[1] = scale[0]
    else:
        scale[0] = scale[1]
    
    # Create transformation matrix
    src_h, src_w = img.shape[:2]
    dst_h, dst_w = 256, 192  # Common size for pose models
    
    # Calculate scaling factor
    scale_factor = min(dst_h / scale[1], dst_w / scale[0])
    
    # Calculate translation to center
    trans = np.array([
        [scale_factor, 0, dst_w / 2 - scale_factor * center[0]],
        [0, scale_factor, dst_h / 2 - scale_factor * center[1]]
    ])
    
    # Apply affine transformation
    transformed_img = cv2.warpAffine(
        img, trans, (dst_w, dst_h), flags=cv2.INTER_LINEAR
    )
    
    return transformed_img, center, scale

def get_max_preds(batch_heatmaps):
    """
    Get predictions from batch heatmaps
    
    Args:
        batch_heatmaps: Batch of heatmaps [batch_size, num_joints, height, width]
        
    Returns:
        Tuple of predicted positions and maxvals
    """
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def nms(heat, kernel=3):
    """
    Non-maximum suppression for heatmaps
    
    Args:
        heat: Heatmap tensor
        kernel: Kernel size for max pooling
        
    Returns:
        Filtered heatmap with only local maxima
    """
    pad = (kernel - 1) // 2
    
    # Find local maximum
    hmax = np.zeros_like(heat)
    for i in range(-pad, pad + 1):
        for j in range(-pad, pad + 1):
            if i == 0 and j == 0:
                continue
            hmax = np.maximum(hmax, np.roll(np.roll(heat, i, axis=0), j, axis=1))
    
    # Only keep pixels that are local maxima
    keep = (heat > hmax)
    
    return heat * keep