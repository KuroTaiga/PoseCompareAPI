import os
import threading
import uuid
import json
from flask import Blueprint, request, jsonify, current_app
import logging

# Import processors
from processors.pose_processor import PoseProcessor
from models.fdhumans_model import FourDHumanWrapper
from models.sapiens_model import SapiensProcessor

logger = logging.getLogger(__name__)

# Create blueprint
process_bp = Blueprint('process', __name__, url_prefix='/api')

# In-memory job storage
# In a production app, use a database or Redis
jobs = {}

def process_video_task(job_id, filename, selected_models, noise_filter, filter_window):
    """Background task for video processing"""
    try:
        jobs[job_id]['status'] = 'processing'
        
        input_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        output_paths = {}
        
        # Initialize required models
        if 'mediapipe' in selected_models:
            processor = PoseProcessor()
        
        if 'fourdhumans' in selected_models:
            fdh_model = FourDHumanWrapper()
        
        # Initialize Sapiens models as needed
        sapiens_processors = {}
        for model in selected_models:
            if model.startswith('sapiens_'):
                model_path = current_app.config['SAPIENS_MODELS'].get(model)
                if model_path:
                    sapiens_processors[model] = SapiensProcessor(model_path, save_img_flag=False)
        
        # Process with each selected model
        for model in selected_models:
            try:
                if model == 'mediapipe':
                    output_path = f'static/results/output_{job_id}_{noise_filter}.mp4'
                    processor.process_video(input_path, method=noise_filter)
                    output_paths[model] = output_path
                
                elif model == 'fourdhumans':
                    output_path = f'static/results/output_{job_id}_{model}.mp4'
                    fdh_model.process_video(input_path, output_path, method=noise_filter)
                    output_paths[model] = output_path
                
                elif model.startswith('sapiens_'):
                    output_path = f'static/results/output_{job_id}_{model}.mp4'
                    sapiens_processors[model].process_video(
                        input_path, 
                        output_path, 
                        method=noise_filter,
                        filter_window=filter_window
                    )
                    output_paths[model] = output_path
                
                logger.info(f"Processed {model} for job {job_id}, saved to {output_path}")
            
            except Exception as e:
                logger.error(f"Error processing {model} for job {job_id}: {str(e)}")
                jobs[job_id]['errors'].append(f"Error processing {model}: {str(e)}")
        
        # Update job status
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['results'] = output_paths
        
        logger.info(f"Job {job_id} completed successfully")
    
    except Exception as e:
        logger.error(f"Error in job {job_id}: {str(e)}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['errors'].append(str(e))

@process_bp.route('/process', methods=['POST'])
def process_video():
    """Start video processing with selected models and filters"""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Required parameters
    filename = data.get('filename')
    selected_models = data.get('models', [])
    
    # Optional parameters with defaults
    noise_filter = data.get('noise_filter', 'original')
    filter_window = int(data.get('filter_window', 5)) 
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    if not selected_models:
        return jsonify({'error': 'No models selected'}), 400
    
    # Check if file exists
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    # Generate job ID if not provided (from upload)
    job_id = data.get('job_id')
    if not job_id:
        job_id = str(uuid.uuid4())
    
    # Create job entry
    jobs[job_id] = {
        'id': job_id,
        'filename': filename,
        'status': 'queued',
        'models': selected_models,
        'noise_filter': noise_filter,
        'filter_window': filter_window,
        'results': {},
        'errors': []
    }
    
    # Start processing in background
    thread = threading.Thread(
        target=process_video_task,
        args=(job_id, filename, selected_models, noise_filter, filter_window)
    )
    thread.daemon = True
    thread.start()
    
    logger.info(f"Started processing job {job_id} for file {filename}")
    
    return jsonify({
        'status': 'success',
        'message': 'Processing started',
        'job_id': job_id
    }), 202