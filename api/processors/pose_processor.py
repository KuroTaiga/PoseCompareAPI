import cv2
import mediapipe as mp
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d, splrep, splev, griddata
from filterpy.kalman import KalmanFilter
from pykrige.ok import OrdinaryKriging
from mediapipe.framework.formats import landmark_pb2
import os
from utils.joint_data import JointDataProcessor

class PoseProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # COCO connections for drawing the skeleton
        # Each tuple represents a connection between two keypoints
        self.coco_connections = [
            (0, 1), (0, 2),  # nose to eyes
            (1, 3), (2, 4),  # eyes to ears
            (5, 6),  # shoulders
            (5, 7), (7, 9),  # left arm
            (6, 8), (8, 10),  # right arm
            (5, 11), (6, 12),  # shoulders to hips
            (11, 12),  # hips
            (11, 13), (13, 15),  # left leg
            (12, 14), (14, 16)  # right leg
        ]

    def process_video(self, input_video_path, method='original', output_path=None, output_json_path=None, sample_rate=1):
        """
        Process a video with the specified method
        
        Args:
            input_video_path (str): Path to the input video
            method (str): Processing method to apply 
            output_path (str, optional): Path to save the output video
            output_json_path (str, optional): Path to save joint data as JSON
            sample_rate (int): Save every Nth frame to JSON (default: 1 = every frame)
                                        
        Returns:
            str: Path to the output video
        """
        print(f"Start processing with {method} method...")
        
        # Set default output paths if not provided
        if output_path is None:
            output_dir = os.path.dirname(input_video_path)
            output_filename = f'output_{method}.mp4'
            output_path = os.path.join(output_dir, output_filename)
            
        if output_json_path is None and method != 'original':
            output_dir = os.path.dirname(output_path)
            output_json_filename = f'joint_data_{method}.json'
            output_json_path = os.path.join(output_dir, output_json_filename)
            
        cap = cv2.VideoCapture(input_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Store landmarks for all frames
        all_landmarks = []
        raw_coco_keypoints = []  # For storing the original COCO keypoints
        frames = []
        frame_indices = []
        frame_count = 0
        max_frames = 1000  # Limit processing to 1000 frames

        # First read all frames and landmarks
        print("Reading and detecting landmarks in frames...")
        while cap.isOpened() and frame_count < max_frames and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Only process every nth frame according to sample_rate
            if frame_count % sample_rate == 0 or method == 'original':
                print(f"\rProcessing frame {frame_count}/{total_frames}", end="")
                    
                frames.append(frame)
                frame_indices.append(frame_count)
                results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if results.pose_landmarks:
                    # Extract landmarks for filtering
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
                    raw_coco_keypoints.append(coco_keypoints)
                else:
                    all_landmarks.append([])
                    # Add placeholder for missing keypoints
                    raw_coco_keypoints.append([[0, 0, 0, 0] for _ in range(17)])

        print(f"\nMediaPipe detection completed, processed {frame_count} frames")
        
        # Convert to numpy array for processing
        all_landmarks = np.array(all_landmarks)
        
        # Process landmarks based on different methods
        if method != 'original' and len(all_landmarks) > 0:
            print(f"Starting to apply {method} method to process landmarks...")
            processed_landmarks = self.apply_method(all_landmarks, method)
            
            # Process COCO keypoints with the same filter for JSON output
            if output_json_path and len(raw_coco_keypoints) > 0:
                processed_coco_keypoints = JointDataProcessor.apply_filter_to_keypoints(
                    raw_coco_keypoints,
                    method
                )
                
                # Save to JSON
                filter_metadata = {
                    "method": method,
                    "video_path": input_video_path,
                    "frame_width": width,
                    "frame_height": height,
                    "fps": fps,
                    "total_frames": total_frames
                }
                
                JointDataProcessor.save_keypoints_to_json(
                    processed_coco_keypoints,
                    frame_indices,
                    output_json_path,
                    metadata=filter_metadata,
                    sample_rate=sample_rate
                )
        else:
            processed_landmarks = all_landmarks
            
            # Save original keypoints to JSON if requested
            if output_json_path and len(raw_coco_keypoints) > 0:
                original_metadata = {
                    "method": "original",
                    "video_path": input_video_path,
                    "frame_width": width,
                    "frame_height": height,
                    "fps": fps,
                    "total_frames": total_frames
                }
                
                JointDataProcessor.save_keypoints_to_json(
                    raw_coco_keypoints,
                    frame_indices,
                    output_json_path,
                    metadata=original_metadata,
                    sample_rate=sample_rate
                )

        # Draw processed results
        print("Starting to generate output video with overlay...")
        for i, frame in enumerate(frames):
            if i < len(processed_landmarks) and len(processed_landmarks[i]) > 0:
                # Convert processed landmarks to MediaPipe format
                landmark_list = landmark_pb2.NormalizedLandmarkList()
                for landmark_data in processed_landmarks[i]:
                    landmark = landmark_list.landmark.add()
                    landmark.x = float(landmark_data[0])
                    landmark.y = float(landmark_data[1])
                    landmark.z = float(landmark_data[2])
                    landmark.visibility = float(landmark_data[3])
                try:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        landmark_list,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                except Exception as e:
                    print(f"Error drawing landmarks: {e}")
            out.write(frame)

        cap.release()
        out.release()
        
        print(f"Completed processing with {method} method")
        print(f"Output video saved to: {output_path}")
        if output_json_path:
            print(f"Joint data saved to: {output_json_path}")
        
        return output_path

    def apply_method(self, landmarks, method):
        """
        Apply a processing method to landmarks
        
        Args:
            landmarks (numpy.ndarray): Array of landmarks
            method (str): Method to apply
            
        Returns:
            numpy.ndarray: Processed landmarks
        """
        if method == 'kalman':
            return self.apply_kalman_filter(landmarks)
        elif method == 'butterworth':
            return self.apply_butterworth_filter(landmarks)
        elif method == 'wiener':
            return self.apply_wiener_filter(landmarks)
        elif method == 'linear':
            return self.apply_linear_interpolation(landmarks)
        elif method == 'bilinear':
            return self.apply_bilinear_interpolation(landmarks)
        elif method == 'spline':
            return self.apply_spline_interpolation(landmarks)
        elif method == 'kriging':
            return self.apply_kriging_interpolation(landmarks)
        elif method == 'chebyshev':
            return self.apply_chebyshev_filter(landmarks)
        elif method == 'bessel':
            return self.apply_bessel_filter(landmarks)
        return landmarks

    def apply_kalman_filter(self, landmarks):
        """
        Apply Kalman filter to smooth landmark trajectories
        
        Args:
            landmarks (numpy.ndarray): Array of landmarks
            
        Returns:
            numpy.ndarray: Filtered landmarks
        """
        print("Applying Kalman filter...")
        filtered_landmarks = np.copy(landmarks)
        
        # Apply Kalman filter to each keypoint
        for point_idx in range(landmarks.shape[1]):
            kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, vx, vy], Measurement: [x, y]
            kf.F = np.array([[1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
            kf.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
            kf.R *= 0.1
            kf.Q *= 0.1
            
            # Get x,y coordinates for current keypoint across all frames
            points = landmarks[:, point_idx, :2]
            filtered_points = []
            
            for point in points:
                kf.predict()
                if point is not None and not np.any(np.isnan(point)):
                    kf.update(point)
                filtered_points.append(kf.x[:2].flatten())  # Ensure 1D array
            
            filtered_points = np.array(filtered_points)  # Convert to numpy array
            filtered_landmarks[:, point_idx, :2] = filtered_points
            filtered_landmarks[:, point_idx, 3] = 1.0  # Set visibility to 1
        
        return filtered_landmarks

    def apply_butterworth_filter(self, landmarks, order=4, cutoff=0.1):
        """
        Apply Butterworth low-pass filter to smooth landmark trajectories
        
        Args:
            landmarks (numpy.ndarray): Array of landmarks
            order (int): Filter order
            cutoff (float): Cutoff frequency (normalized between 0 and 1)
            
        Returns:
            numpy.ndarray: Filtered landmarks
        """
        print("Applying Butterworth filter...")
        filtered_landmarks = np.copy(landmarks)
        
        # Design Butterworth filter
        b, a = signal.butter(order, cutoff, 'low')
        
        # Apply filter to each keypoint
        for point_idx in range(landmarks.shape[1]):
            for dim in range(2):  # Only filter x,y coordinates
                signal_1d = landmarks[:, point_idx, dim]
                # Apply filter
                filtered_signal = signal.filtfilt(b, a, signal_1d)
                filtered_landmarks[:, point_idx, dim] = filtered_signal
            filtered_landmarks[:, point_idx, 3] = 1.0  # Set visibility to 1
        
        return filtered_landmarks

    def apply_wiener_filter(self, landmarks, mysize=5):
        """
        Apply Wiener filter to reduce noise in landmark trajectories
        
        Args:
            landmarks (numpy.ndarray): Array of landmarks
            mysize (int): Size of the filter window
            
        Returns:
            numpy.ndarray: Filtered landmarks
        """
        print("Applying Wiener filter...")
        filtered_landmarks = np.copy(landmarks)
        
        # Apply Wiener filter to each keypoint
        for point_idx in range(landmarks.shape[1]):
            for dim in range(2):  # Only filter x,y coordinates
                signal_1d = landmarks[:, point_idx, dim]
                # Apply Wiener filter
                filtered_signal = signal.wiener(signal_1d, mysize=mysize)
                filtered_landmarks[:, point_idx, dim] = filtered_signal
            filtered_landmarks[:, point_idx, 3] = 1.0  # Set visibility to 1
        
        return filtered_landmarks

    def apply_linear_interpolation(self, landmarks):
        """
        Apply linear interpolation to fill in missing landmarks
        
        Args:
            landmarks (numpy.ndarray): Array of landmarks
            
        Returns:
            numpy.ndarray: Interpolated landmarks
        """
        print("Applying linear interpolation...")
        interpolated_landmarks = np.copy(landmarks)
        
        # Interpolate each keypoint
        for point_idx in range(landmarks.shape[1]):
            # Get valid frame indices (non-None frames)
            valid_frames = np.where(landmarks[:, point_idx, 3] > 0.5)[0]
            if len(valid_frames) < 2:
                continue
                
            # Interpolate x,y,z coordinates separately
            for dim in range(3):
                f = interp1d(valid_frames, 
                           landmarks[valid_frames, point_idx, dim],
                           kind='linear',
                           fill_value="extrapolate")
                interpolated_landmarks[:, point_idx, dim] = f(np.arange(len(landmarks)))
            
            interpolated_landmarks[:, point_idx, 3] = 1.0  # Set visibility to 1
        
        return interpolated_landmarks

    def apply_spline_interpolation(self, landmarks):
        """
        Apply cubic spline interpolation to fill in missing landmarks
        
        Args:
            landmarks (numpy.ndarray): Array of landmarks
            
        Returns:
            numpy.ndarray: Interpolated landmarks
        """
        print("Applying spline interpolation...")
        interpolated_landmarks = np.copy(landmarks)
        
        # Interpolate each keypoint
        for point_idx in range(landmarks.shape[1]):
            # Get valid frame indices
            valid_frames = np.where(landmarks[:, point_idx, 3] > 0.5)[0]
            if len(valid_frames) < 4:  # Spline interpolation needs at least 4 points
                continue
                
            # Interpolate x,y,z coordinates separately
            for dim in range(3):
                # Use cubic spline interpolation
                tck = splrep(valid_frames, 
                           landmarks[valid_frames, point_idx, dim],
                           k=3)
                interpolated_landmarks[:, point_idx, dim] = splev(np.arange(len(landmarks)), tck)
            
            interpolated_landmarks[:, point_idx, 3] = 1.0  # Set visibility to 1
        
        return interpolated_landmarks

    def apply_bilinear_interpolation(self, landmarks):
        """
        Apply bilinear interpolation to fill in missing landmarks
        
        Args:
            landmarks (numpy.ndarray): Array of landmarks
            
        Returns:
            numpy.ndarray: Interpolated landmarks
        """
        print("Applying bilinear interpolation...")
        interpolated_landmarks = np.copy(landmarks)
        
        # Interpolate each keypoint
        for point_idx in range(landmarks.shape[1]):
            # Get valid frame indices
            valid_frames = np.where(landmarks[:, point_idx, 3] > 0.5)[0]
            if len(valid_frames) < 2:
                continue
                
            # Create time grid
            time_grid = np.arange(len(landmarks))
            
            # Apply bilinear interpolation to x,y coordinates
            for dim in range(2):
                interpolated_landmarks[:, point_idx, dim] = griddata(
                    valid_frames, 
                    landmarks[valid_frames, point_idx, dim],
                    time_grid,
                    method='linear'
                )
            
            interpolated_landmarks[:, point_idx, 3] = 1.0  # Set visibility to 1
        
        return interpolated_landmarks

    def apply_kriging_interpolation(self, landmarks):
        """
        Apply Kriging interpolation to fill in missing landmarks
        
        Args:
            landmarks (numpy.ndarray): Array of landmarks
            
        Returns:
            numpy.ndarray: Interpolated landmarks
        """
        print("Applying Kriging interpolation...")
        interpolated_landmarks = np.copy(landmarks)
        
        # Interpolate each keypoint
        for point_idx in range(landmarks.shape[1]):
            # Get valid frame indices
            valid_frames = np.where(landmarks[:, point_idx, 3] > 0.5)[0]
            if len(valid_frames) < 3:  # Kriging needs at least 3 points
                continue
            
            # Apply Kriging interpolation to x,y coordinates
            for dim in range(2):
                try:
                    # Prepare data - ensure float64 type
                    x = valid_frames.astype(np.float64).reshape(-1, 1)  # Time as x coordinate
                    y = np.zeros_like(x)  # Virtual y coordinate
                    z = landmarks[valid_frames, point_idx, dim].astype(np.float64)  # Actual values
                    
                    # Create Kriging model
                    ok = OrdinaryKriging(
                        x.flatten(),
                        y.flatten(),
                        z,
                        variogram_model='gaussian',
                        verbose=False,
                        enable_plotting=False
                    )
                    
                    # Make predictions
                    grid_x = np.arange(len(landmarks), dtype=np.float64).reshape(-1, 1)
                    grid_y = np.zeros_like(grid_x)
                    z_pred, _ = ok.execute('points', grid_x.flatten(), grid_y.flatten())
                    
                    interpolated_landmarks[:, point_idx, dim] = z_pred
                except Exception as e:
                    print(f"Kriging interpolation failed, using spline interpolation instead: {e}")
                    if len(valid_frames) >= 4:
                        tck = splrep(valid_frames, landmarks[valid_frames, point_idx, dim], k=3)
                        interpolated_landmarks[:, point_idx, dim] = splev(np.arange(len(landmarks)), tck)
                    else:
                        f = interp1d(valid_frames, 
                                   landmarks[valid_frames, point_idx, dim],
                                   kind='linear',
                                   fill_value="extrapolate")
                        interpolated_landmarks[:, point_idx, dim] = f(np.arange(len(landmarks)))
            
            # Use spline interpolation for z coordinate
            if len(valid_frames) >= 4:
                tck = splrep(valid_frames, landmarks[valid_frames, point_idx, 2], k=3)
                interpolated_landmarks[:, point_idx, 2] = splev(np.arange(len(landmarks)), tck)
            else:
                f = interp1d(valid_frames, 
                           landmarks[valid_frames, point_idx, 2],
                           kind='linear',
                           fill_value="extrapolate")
                interpolated_landmarks[:, point_idx, 2] = f(np.arange(len(landmarks)))
            
            # Set visibility to 1 to indicate all points are visible
            interpolated_landmarks[:, point_idx, 3] = 1.0
        
        return interpolated_landmarks
    
    def apply_chebyshev_filter(self, landmarks, order=4, ripple_db=1.0, cutoff=0.1):
        """
        Apply Chebyshev Type I filter to smooth landmark trajectories
        
        Args:
            landmarks (numpy.ndarray): Array of landmarks
            order (int): Filter order
            ripple_db (float): Maximum ripple allowed in the passband (dB)
            cutoff (float): Cutoff frequency (normalized between 0 and 1)
            
        Returns:
            numpy.ndarray: Filtered landmarks
        """
        print("Applying Chebyshev filter...")
        filtered_landmarks = np.copy(landmarks)
        
        # Design Chebyshev filter
        b, a = signal.cheby1(order, ripple_db, cutoff, 'low')
        
        # Apply filter to each keypoint
        for point_idx in range(landmarks.shape[1]):
            for dim in range(2):  # Only filter x,y coordinates
                signal_1d = landmarks[:, point_idx, dim]
                # Apply filter
                filtered_signal = signal.filtfilt(b, a, signal_1d)
                filtered_landmarks[:, point_idx, dim] = filtered_signal
            filtered_landmarks[:, point_idx, 3] = 1.0  # Set visibility to 1
        
        return filtered_landmarks

    def apply_bessel_filter(self, landmarks, order=4, cutoff=0.1):
        """
        Apply Bessel filter to smooth landmark trajectories
        
        Args:
            landmarks (numpy.ndarray): Array of landmarks
            order (int): Filter order
            cutoff (float): Cutoff frequency (normalized between 0 and 1)
            
        Returns:
            numpy.ndarray: Filtered landmarks
        """
        print("Applying Bessel filter...")
        filtered_landmarks = np.copy(landmarks)
        
        # Design Bessel filter
        b, a = signal.bessel(order, cutoff, 'low')
        
        # Apply filter to each keypoint
        for point_idx in range(landmarks.shape[1]):
            for dim in range(2):  # Only filter x,y coordinates
                signal_1d = landmarks[:, point_idx, dim]
                # Apply filter
                filtered_signal = signal.filtfilt(b, a, signal_1d)
                filtered_landmarks[:, point_idx, dim] = filtered_signal
            filtered_landmarks[:, point_idx, 3] = 1.0  # Set visibility to 1
        
        return filtered_landmarks