# Pose Processing API

A Flask-based REST API for processing videos with various pose estimation models and applying filters to keypoint data.

## Overview

This API allows users to:
- Upload videos for pose estimation
- Process videos using different models (MediaPipe, Sapiens)
- Apply various noise filters and interpolation methods to keypoint data
- Visualize pose estimation results
- Download processed videos and extracted keypoint data

## Features

- **Multiple Pose Estimation Models**:
  - MediaPipe Pose (Google)
  - Sapiens models (varying sizes: 0.3B, 0.6B, 1B, 2B)

- **Filtering Methods**:
  - Original (no filter)
  - Butterworth
  - Chebyshev
  - Bessel

- **Interpolation Methods**:
  - Kalman filter
  - Wiener filter
  - Linear interpolation
  - Spline interpolation
  - Bilinear interpolation
  - Kriging interpolation

- **Asynchronous Processing**:
  - Background job processing
  - Job status monitoring

- **Session Management**:
  - User session tracking
  - Automatic cleanup of expired sessions

## Installation

### Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended for Sapiens models)
- FFmpeg (for video processing)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pose-processing-api.git
   cd pose-processing-api
   ```

2. Run the setup script to create the environment and download models:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Activate the conda environment:
   ```bash
   conda activate poseapi
   ```

4. Run the API:
   ```bash
   cd api
   python run.py
   ```

## API Endpoints

### Session Management

- **GET /api/session**
  - Get current session or create a new one
  - Returns session ID and metadata

### File Management

- **POST /api/upload**
  - Upload a video file for processing
  - Supports MP4, AVI, MOV, and WEBM formats
  - Maximum file size: 100MB

- **GET /api/files**
  - Get all files for the current session

- **GET /api/upload/{upload_id}/files**
  - Get all files for a specific upload

### Processing

- **POST /api/process**
  - Process a video with selected models and filters
  - Parameters:
    - `upload_id`: ID of the uploaded video
    - `models`: Array of model names to use
    - `noise_filter`: Filter method to apply
    - `filter_window`: Filter window size
    - `sample_rate`: Keypoint sampling rate

- **GET /api/process/available-models**
  - Get a list of available models and filters

### Jobs

- **GET /api/jobs/{job_id}**
  - Get the status of a specific job

- **GET /api/jobs/{job_id}/result**
  - Get the results of a completed job

- **GET /api/jobs/{job_id}/download**
  - Download all results from a job

- **DELETE /api/jobs/{job_id}**
  - Delete a job (doesn't delete result files)

### Info

- **GET /api/info**
  - Get API information and status

## Usage Example

1. Start a session:
   ```bash
   curl -X GET http://localhost:5000/api/session
   ```

2. Upload a video:
   ```bash
   curl -X POST http://localhost:5000/api/upload -F "file=@path/to/video.mp4"
   ```

3. Process the video:
   ```bash
   curl -X POST http://localhost:5000/api/process \
     -H "Content-Type: application/json" \
     -d '{
       "upload_id": "YOUR_UPLOAD_ID",
       "models": ["mediapipe", "sapiens_1b"],
       "noise_filter": "butterworth",
       "filter_window": 5,
       "sample_rate": 1
     }'
   ```

4. Check job status:
   ```bash
   curl -X GET http://localhost:5000/api/jobs/YOUR_JOB_ID
   ```

5. Get results when job is completed:
   ```bash
   curl -X GET http://localhost:5000/api/jobs/YOUR_JOB_ID/result
   ```

## Storage Structure

```
/storage/
    /sessions/
        /<session_id>/
            metadata.json
            /<upload_id>/
                original.mp4  # Original uploaded video
                mediapipe_butterworth.mp4  # Processed output
                mediapipe_butterworth.json  # Keypoint data
                sapiens_1b_butterworth.mp4  # Processed output
                sapiens_1b_butterworth.json  # Keypoint data
                ... other processing results
```

## Keypoint Data Format

The API exports keypoint data in the COCO-17 format as JSON:

```json
{
  "metadata": {
    "model": "mediapipe",
    "method": "butterworth",
    "filter_window": 5,
    "video_path": "path/to/video.mp4",
    "frame_width": 1280,
    "frame_height": 720,
    "fps": 30,
    "total_frames": 300
  },
  "frame_count": 300,
  "keypoint_names": ["nose", "left_eye", "right_eye", ...],
  "sample_rate": 1,
  "frames": [
    {
      "index": 1,
      "keypoints": [
        {
          "name": "nose",
          "position": {"x": 640.5, "y": 380.2, "z": 0.0},
          "confidence": 0.98
        },
        // ... more keypoints
      ]
    },
    // ... more frames
  ]
}
```

## License

[Your License Information]

## Credits

- MediaPipe by Google
- Sapiens Pose Estimation Model by Facebook/Meta