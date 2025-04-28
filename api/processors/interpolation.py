"""
Interpolation methods for pose data

This module provides various interpolation methods that can be applied
to keypoint trajectories from pose estimation models, particularly for
filling in missing or low-confidence keypoints.
"""

import numpy as np
from scipy.interpolate import interp1d, splrep, splev, griddata
from pykrige.ok import OrdinaryKriging

class PoseInterpolation:
    """
    Class for applying various interpolation methods to pose keypoint data
    """
    
    @staticmethod
    def apply_interpolation(keypoints, method, **kwargs):
        """
        Apply an interpolation method to keypoint data
        
        Args:
            keypoints (numpy.ndarray): Array of keypoints [frames, keypoints, coords]
            method (str): Interpolation method to apply
            **kwargs: Additional parameters for specific interpolation methods
            
        Returns:
            numpy.ndarray: Interpolated keypoints
        """
        # Create a copy to avoid modifying the original
        interpolated_keypoints = np.copy(keypoints)
        
        if method == 'no interpolation':
            return interpolated_keypoints
        
        # Apply the selected interpolation method
        if method == 'linear':
            return PoseInterpolation.linear_interpolation(interpolated_keypoints, **kwargs)
        elif method == 'spline':
            return PoseInterpolation.spline_interpolation(interpolated_keypoints, **kwargs)
        elif method == 'bilinear':
            return PoseInterpolation.bilinear_interpolation(interpolated_keypoints, **kwargs)
        elif method == 'kriging':
            return PoseInterpolation.kriging_interpolation(interpolated_keypoints, **kwargs)
        else:
            print(f"Warning: Unknown interpolation method '{method}', returning original data")
            return interpolated_keypoints
    
    @staticmethod
    def linear_interpolation(keypoints, confidence_threshold=0.5, **kwargs):
        """
        Apply linear interpolation to fill in missing keypoints
        
        Args:
            keypoints (numpy.ndarray): Array of keypoints [frames, keypoints, coords]
            confidence_threshold (float): Threshold for considering keypoints valid
            
        Returns:
            numpy.ndarray: Interpolated keypoints
        """
        print("Applying linear interpolation...")
        interpolated_keypoints = np.copy(keypoints)
        
        # Determine if we have confidence values (4-element keypoints [x,y,z,confidence])
        has_confidence = keypoints.shape[2] > 3
        
        # Interpolate each keypoint
        for point_idx in range(keypoints.shape[1]):
            # Get valid frame indices based on confidence or existence
            if has_confidence:
                valid_frames = np.where(keypoints[:, point_idx, 3] > confidence_threshold)[0]
            else:
                # If no confidence values, assume all frames are valid
                valid_frames = np.arange(len(keypoints))
            
            if len(valid_frames) < 2:
                continue  # Need at least 2 points for interpolation
                
            # Interpolate coordinates separately
            for dim in range(min(3, keypoints.shape[2])):
                f = interp1d(valid_frames, 
                           keypoints[valid_frames, point_idx, dim],
                           kind='linear',
                           fill_value="extrapolate")
                interpolated_keypoints[:, point_idx, dim] = f(np.arange(len(keypoints)))
            
            # Set confidence to 1.0 if we have confidence values
            if has_confidence:
                interpolated_keypoints[:, point_idx, 3] = 1.0
        
        return interpolated_keypoints
    
    @staticmethod
    def spline_interpolation(keypoints, confidence_threshold=0.5, k=3, **kwargs):
        """
        Apply cubic spline interpolation to fill in missing keypoints
        
        Args:
            keypoints (numpy.ndarray): Array of keypoints [frames, keypoints, coords]
            confidence_threshold (float): Threshold for considering keypoints valid
            k (int): Degree of the spline (1=linear, 2=quadratic, 3=cubic)
            
        Returns:
            numpy.ndarray: Interpolated keypoints
        """
        print("Applying spline interpolation...")
        interpolated_keypoints = np.copy(keypoints)
        
        # Determine if we have confidence values
        has_confidence = keypoints.shape[2] > 3
        
        # Interpolate each keypoint
        for point_idx in range(keypoints.shape[1]):
            # Get valid frame indices
            if has_confidence:
                valid_frames = np.where(keypoints[:, point_idx, 3] > confidence_threshold)[0]
            else:
                valid_frames = np.arange(len(keypoints))
                
            # Spline interpolation needs at least k+1 points
            if len(valid_frames) < k+1:
                # Fall back to linear interpolation if not enough points
                if len(valid_frames) >= 2:
                    for dim in range(min(3, keypoints.shape[2])):
                        f = interp1d(valid_frames, 
                                   keypoints[valid_frames, point_idx, dim],
                                   kind='linear',
                                   fill_value="extrapolate")
                        interpolated_keypoints[:, point_idx, dim] = f(np.arange(len(keypoints)))
                continue
                
            # Interpolate coordinates separately
            for dim in range(min(3, keypoints.shape[2])):
                # Use spline interpolation
                tck = splrep(valid_frames, 
                           keypoints[valid_frames, point_idx, dim],
                           k=k)
                interpolated_keypoints[:, point_idx, dim] = splev(np.arange(len(keypoints)), tck)
            
            # Set confidence to 1.0 if applicable
            if has_confidence:
                interpolated_keypoints[:, point_idx, 3] = 1.0
        
        return interpolated_keypoints
    
    @staticmethod
    def bilinear_interpolation(keypoints, confidence_threshold=0.5, **kwargs):
        """
        Apply bilinear interpolation to fill in missing keypoints
        
        Args:
            keypoints (numpy.ndarray): Array of keypoints [frames, keypoints, coords]
            confidence_threshold (float): Threshold for considering keypoints valid
            
        Returns:
            numpy.ndarray: Interpolated keypoints
        """
        print("Applying bilinear interpolation...")
        interpolated_keypoints = np.copy(keypoints)
        
        # Determine if we have confidence values
        has_confidence = keypoints.shape[2] > 3
        
        # Interpolate each keypoint
        for point_idx in range(keypoints.shape[1]):
            # Get valid frame indices
            if has_confidence:
                valid_frames = np.where(keypoints[:, point_idx, 3] > confidence_threshold)[0]
            else:
                valid_frames = np.arange(len(keypoints))
                
            if len(valid_frames) < 2:
                continue  # Need at least 2 points for interpolation
                
            # Create time grid
            time_grid = np.arange(len(keypoints))
            
            # Apply bilinear interpolation to coordinates
            for dim in range(min(3, keypoints.shape[2])):
                try:
                    # Use griddata with linear method (equivalent to bilinear for 1D case)
                    interpolated_keypoints[:, point_idx, dim] = griddata(
                        valid_frames, 
                        keypoints[valid_frames, point_idx, dim],
                        time_grid,
                        method='linear'
                    )
                    
                    # Handle potential NaN values from extrapolation
                    mask = np.isnan(interpolated_keypoints[:, point_idx, dim])
                    if np.any(mask):
                        # Use nearest neighbor for extrapolation
                        nn_values = griddata(
                            valid_frames, 
                            keypoints[valid_frames, point_idx, dim],
                            time_grid[mask],
                            method='nearest'
                        )
                        interpolated_keypoints[mask, point_idx, dim] = nn_values
                except Exception as e:
                    print(f"Error in bilinear interpolation for point {point_idx}, dimension {dim}: {e}")
                    # Fall back to linear interpolation on error
                    try:
                        f = interp1d(valid_frames, 
                                   keypoints[valid_frames, point_idx, dim],
                                   kind='linear',
                                   fill_value="extrapolate")
                        interpolated_keypoints[:, point_idx, dim] = f(np.arange(len(keypoints)))
                    except Exception as e2:
                        print(f"Linear fallback also failed: {e2}")
            
            # Set confidence to 1.0 if applicable
            if has_confidence:
                interpolated_keypoints[:, point_idx, 3] = 1.0
        
        return interpolated_keypoints
    
    @staticmethod
    def kriging_interpolation(keypoints, confidence_threshold=0.5, 
                              variogram_model='gaussian', verbose=False, **kwargs):
        """
        Apply Kriging interpolation to fill in missing keypoints
        
        Args:
            keypoints (numpy.ndarray): Array of keypoints [frames, keypoints, coords]
            confidence_threshold (float): Threshold for considering keypoints valid
            variogram_model (str): Variogram model ('linear', 'gaussian', etc.)
            verbose (bool): Whether to print verbose output
            
        Returns:
            numpy.ndarray: Interpolated keypoints
        """
        print("Applying Kriging interpolation...")
        interpolated_keypoints = np.copy(keypoints)
        
        # Determine if we have confidence values
        has_confidence = keypoints.shape[2] > 3
        
        # Interpolate each keypoint
        for point_idx in range(keypoints.shape[1]):
            # Get valid frame indices
            if has_confidence:
                valid_frames = np.where(keypoints[:, point_idx, 3] > confidence_threshold)[0]
            else:
                valid_frames = np.arange(len(keypoints))
                
            if len(valid_frames) < 3:  # Kriging needs at least 3 points
                # Fall back to spline or linear if insufficient points
                if len(valid_frames) >= 2:
                    for dim in range(min(3, keypoints.shape[2])):
                        f = interp1d(valid_frames, 
                                   keypoints[valid_frames, point_idx, dim],
                                   kind='linear',
                                   fill_value="extrapolate")
                        interpolated_keypoints[:, point_idx, dim] = f(np.arange(len(keypoints)))
                continue
            
            # Apply Kriging interpolation to x,y coordinates
            for dim in range(2):  # Typically only apply to x,y, not z
                try:
                    # Prepare data - ensure float64 type
                    x = valid_frames.astype(np.float64).reshape(-1, 1)  # Time as x coordinate
                    y = np.zeros_like(x)  # Virtual y coordinate
                    z = keypoints[valid_frames, point_idx, dim].astype(np.float64)  # Actual values
                    
                    # Create Kriging model
                    ok = OrdinaryKriging(
                        x.flatten(),
                        y.flatten(),
                        z,
                        variogram_model=variogram_model,
                        verbose=verbose,
                        enable_plotting=False
                    )
                    
                    # Make predictions
                    grid_x = np.arange(len(keypoints), dtype=np.float64).reshape(-1, 1)
                    grid_y = np.zeros_like(grid_x)
                    z_pred, _ = ok.execute('points', grid_x.flatten(), grid_y.flatten())
                    
                    interpolated_keypoints[:, point_idx, dim] = z_pred
                except Exception as e:
                    print(f"Kriging interpolation failed for point {point_idx}, dimension {dim}: {e}")
                    # Fall back to spline interpolation
                    try:
                        if len(valid_frames) >= 4:
                            tck = splrep(valid_frames, keypoints[valid_frames, point_idx, dim], k=3)
                            interpolated_keypoints[:, point_idx, dim] = splev(np.arange(len(keypoints)), tck)
                        else:
                            f = interp1d(valid_frames, 
                                       keypoints[valid_frames, point_idx, dim],
                                       kind='linear',
                                       fill_value="extrapolate")
                            interpolated_keypoints[:, point_idx, dim] = f(np.arange(len(keypoints)))
                    except Exception as e2:
                        print(f"Fallback interpolation also failed: {e2}")
            
            # Handle z coordinate with spline or linear interpolation
            if keypoints.shape[2] > 2:
                try:
                    if len(valid_frames) >= 4:
                        tck = splrep(valid_frames, keypoints[valid_frames, point_idx, 2], k=3)
                        interpolated_keypoints[:, point_idx, 2] = splev(np.arange(len(keypoints)), tck)
                    else:
                        f = interp1d(valid_frames, 
                                   keypoints[valid_frames, point_idx, 2],
                                   kind='linear',
                                   fill_value="extrapolate")
                        interpolated_keypoints[:, point_idx, 2] = f(np.arange(len(keypoints)))
                except Exception as e:
                    print(f"Z-coordinate interpolation failed for point {point_idx}: {e}")
            
            # Set confidence to 1.0 if applicable
            if has_confidence:
                interpolated_keypoints[:, point_idx, 3] = 1.0
        
        return interpolated_keypoints