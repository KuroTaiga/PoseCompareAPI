"""
Noise filtering module for pose data

This module provides various noise filtering methods that can be applied
to keypoint trajectories from pose estimation models.
"""

import numpy as np
from scipy import signal
from filterpy.kalman import KalmanFilter

class PoseFilter:
    """
    Class for applying various noise filters to pose keypoint data
    """
    
    @staticmethod
    def apply_filter(keypoints, method, filter_window=5, **kwargs):
        """
        Apply a filtering method to keypoint data
        
        Args:
            keypoints (numpy.ndarray): Array of keypoints [frames, keypoints, coords]
            method (str): Filtering method to apply
            filter_window (int): Size of filtering window
            **kwargs: Additional parameters for specific filters
            
        Returns:
            numpy.ndarray: Filtered keypoints
        """
        # Create a copy to avoid modifying the original
        filtered_keypoints = np.copy(keypoints)
        
        if method == 'original':
            return filtered_keypoints
        
        # Apply the selected filter
        if method == 'butterworth':
            return PoseFilter.butterworth_filter(filtered_keypoints, **kwargs)
        elif method == 'kalman':
            return PoseFilter.kalman_filter(filtered_keypoints, **kwargs)
        elif method == 'wiener':
            return PoseFilter.wiener_filter(filtered_keypoints, mysize=filter_window, **kwargs)
        elif method == 'chebyshev':
            return PoseFilter.chebyshev_filter(filtered_keypoints, **kwargs)
        elif method == 'bessel':
            return PoseFilter.bessel_filter(filtered_keypoints, **kwargs)
        else:
            print(f"Warning: Unknown filter method '{method}', returning original data")
            return filtered_keypoints
    
    @staticmethod
    def butterworth_filter(keypoints, order=4, cutoff=0.1, **kwargs):
        """
        Apply Butterworth low-pass filter to smooth keypoint trajectories
        
        Args:
            keypoints (numpy.ndarray): Array of keypoints [frames, keypoints, coords]
            order (int): Filter order
            cutoff (float): Cutoff frequency (normalized between 0 and 1)
            
        Returns:
            numpy.ndarray: Filtered keypoints
        """
        print("Applying Butterworth filter...")
        filtered_keypoints = np.copy(keypoints)
        
        # Design Butterworth filter
        b, a = signal.butter(order, cutoff, 'low')
        
        # Apply filter to each keypoint
        for point_idx in range(keypoints.shape[1]):
            for dim in range(2):  # Only filter x,y coordinates
                signal_1d = keypoints[:, point_idx, dim]
                # Apply filter
                filtered_signal = signal.filtfilt(b, a, signal_1d)
                filtered_keypoints[:, point_idx, dim] = filtered_signal
            
            # Set visibility/confidence to 1 if using 4-element keypoints [x,y,z,confidence]
            if keypoints.shape[2] > 3:
                filtered_keypoints[:, point_idx, 3] = 1.0
        
        return filtered_keypoints
    
    @staticmethod
    def kalman_filter(keypoints, process_noise_scale=0.1, measurement_noise_scale=0.1, **kwargs):
        """
        Apply Kalman filter to smooth keypoint trajectories
        
        Args:
            keypoints (numpy.ndarray): Array of keypoints [frames, keypoints, coords]
            process_noise_scale (float): Process noise scale factor
            measurement_noise_scale (float): Measurement noise scale factor
            
        Returns:
            numpy.ndarray: Filtered keypoints
        """
        print("Applying Kalman filter...")
        filtered_keypoints = np.copy(keypoints)
        
        # Apply Kalman filter to each keypoint
        for point_idx in range(keypoints.shape[1]):
            kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, vx, vy], Measurement: [x, y]
            kf.F = np.array([[1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
            kf.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
            kf.R *= measurement_noise_scale
            kf.Q *= process_noise_scale
            
            # Get x,y coordinates for current keypoint across all frames
            points = keypoints[:, point_idx, :2]
            filtered_points = []
            
            for point in points:
                kf.predict()
                if point is not None and not np.any(np.isnan(point)):
                    kf.update(point)
                filtered_points.append(kf.x[:2].flatten())  # Ensure 1D array
            
            filtered_points = np.array(filtered_points)  # Convert to numpy array
            filtered_keypoints[:, point_idx, :2] = filtered_points
            
            # Set visibility/confidence to 1 if using 4-element keypoints [x,y,z,confidence]
            if keypoints.shape[2] > 3:
                filtered_keypoints[:, point_idx, 3] = 1.0
        
        return filtered_keypoints
    
    @staticmethod
    def wiener_filter(keypoints, mysize=5, **kwargs):
        """
        Apply Wiener filter to reduce noise in keypoint trajectories
        
        Args:
            keypoints (numpy.ndarray): Array of keypoints [frames, keypoints, coords]
            mysize (int): Size of the filter window
            
        Returns:
            numpy.ndarray: Filtered keypoints
        """
        print("Applying Wiener filter...")
        filtered_keypoints = np.copy(keypoints)
        
        # Apply Wiener filter to each keypoint
        for point_idx in range(keypoints.shape[1]):
            for dim in range(2):  # Only filter x,y coordinates
                signal_1d = keypoints[:, point_idx, dim]
                # Apply Wiener filter
                filtered_signal = signal.wiener(signal_1d, mysize=mysize)
                filtered_keypoints[:, point_idx, dim] = filtered_signal
            
            # Set visibility/confidence to 1 if using 4-element keypoints [x,y,z,confidence]
            if keypoints.shape[2] > 3:
                filtered_keypoints[:, point_idx, 3] = 1.0
        
        return filtered_keypoints
    
    @staticmethod
    def chebyshev_filter(keypoints, order=4, ripple_db=1.0, cutoff=0.1, **kwargs):
        """
        Apply Chebyshev Type I filter to smooth keypoint trajectories
        
        Args:
            keypoints (numpy.ndarray): Array of keypoints [frames, keypoints, coords]
            order (int): Filter order
            ripple_db (float): Maximum ripple allowed in the passband (dB)
            cutoff (float): Cutoff frequency (normalized between 0 and 1)
            
        Returns:
            numpy.ndarray: Filtered keypoints
        """
        print("Applying Chebyshev filter...")
        filtered_keypoints = np.copy(keypoints)
        
        # Design Chebyshev filter
        b, a = signal.cheby1(order, ripple_db, cutoff, 'low')
        
        # Apply filter to each keypoint
        for point_idx in range(keypoints.shape[1]):
            for dim in range(2):  # Only filter x,y coordinates
                signal_1d = keypoints[:, point_idx, dim]
                # Apply filter
                filtered_signal = signal.filtfilt(b, a, signal_1d)
                filtered_keypoints[:, point_idx, dim] = filtered_signal
            
            # Set visibility/confidence to 1 if using 4-element keypoints [x,y,z,confidence]
            if keypoints.shape[2] > 3:
                filtered_keypoints[:, point_idx, 3] = 1.0
        
        return filtered_keypoints
    
    @staticmethod
    def bessel_filter(keypoints, order=4, cutoff=0.1, **kwargs):
        """
        Apply Bessel filter to smooth keypoint trajectories
        
        Args:
            keypoints (numpy.ndarray): Array of keypoints [frames, keypoints, coords]
            order (int): Filter order
            cutoff (float): Cutoff frequency (normalized between 0 and 1)
            
        Returns:
            numpy.ndarray: Filtered keypoints
        """
        print("Applying Bessel filter...")
        filtered_keypoints = np.copy(keypoints)
        
        # Design Bessel filter
        b, a = signal.bessel(order, cutoff, 'low')
        
        # Apply filter to each keypoint
        for point_idx in range(keypoints.shape[1]):
            for dim in range(2):  # Only filter x,y coordinates
                signal_1d = keypoints[:, point_idx, dim]
                # Apply filter
                filtered_signal = signal.filtfilt(b, a, signal_1d)
                filtered_keypoints[:, point_idx, dim] = filtered_signal
            
            # Set visibility/confidence to 1 if using 4-element keypoints [x,y,z,confidence]
            if keypoints.shape[2] > 3:
                filtered_keypoints[:, point_idx, 3] = 1.0
        
        return filtered_keypoints