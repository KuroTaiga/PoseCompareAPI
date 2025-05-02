import os
import threading
import uuid
import torch
from flask import Blueprint, request, jsonify, current_app, session, url_for

# from models.fdhumans_model import FourDHumanWrapper
from models.sapiens.model import SapiensProcessor
from utils.file_manager import FileManager

# Create blueprint
process_bp = Blueprint('process', __name__, url_prefix='/api')

# In-memory job storage
# In a production app, use a database or Redis
jobs = {}

def process_video_task(app, job_id, session_id, upload_id, input_path, selected_models, noise_filter, filter_window, sample_rate):
    """Background task for video processing"""
    with app.app_context():
        try:
            jobs[job_id]['status'] = 'processing'
            # Initialize file manager
            file_manager = FileManager(current_app.config['STORAGE_BASE_DIR'])
            
            # Get upload directory
            upload_dir, error = file_manager.get_upload_directory(session_id, upload_id)
            if not upload_dir:
                jobs[job_id]['status'] = 'failed'
                jobs[job_id]['errors'].append(f"Error getting upload directory: {error}")
                return
            
            # Process with each selected model
            for model in selected_models:
                try:
                    print(f"Processing {model} for job {job_id}")                  
                    process_with_model(model, app, job_id, session_id, upload_id, input_path, upload_dir, noise_filter, filter_window, sample_rate, file_manager)
                    
                    # Force garbage collection after each model process
                    import gc
                    gc.collect()

                    #If using CUDA, empty cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()                    
                except Exception as e:
                    print(f"Error processing {model} for job {job_id}: {str(e)}")
                    jobs[job_id]['errors'].append(f"Error processing {model}: {str(e)}")
            
            # Update job status
            jobs[job_id]['status'] = 'completed'
            
            print(f"Job {job_id} completed successfully")
        
        except Exception as e:
            print(f"Error in job {job_id}: {str(e)}")
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['errors'].append(str(e))

def process_with_model(model, app, job_id, session_id, upload_id, input_path, upload_dir, noise_filter, filter_window, sample_rate, file_manager):
    """Process a specific model and handle resources properly"""
    
    if model == 'mediapipe':
        # Create output paths directly in the upload directory
        output_video_path = os.path.join(upload_dir, f'mediapipe_{noise_filter}.mp4')
        output_json_path = os.path.join(upload_dir, f'mediapipe_{noise_filter}.json')
        
        # Initialize processor
        from models.mediapipe_model import MediaPipeProcessor
        processor = MediaPipeProcessor()
        
        # Process the video
        result = processor.process_video(
            input_path, 
            output_video_path,
            method=noise_filter, 
            filter_window=filter_window,
            output_json_path=output_json_path,
            sample_rate=sample_rate
        )
        
        if result:
            # Add to job results
            jobs[job_id]['results'][model] = {
                'video': output_video_path,
                'json': output_json_path
            }
            
            # Add to session metadata for video
            file_manager.save_result(
                session_id, 
                upload_id, 
                f'mediapipe_{noise_filter}', 
                output_video_path, 
                metadata={
                    'model': 'mediapipe',
                    'filter': noise_filter,
                    'window': filter_window,
                    'type': 'video'
                }
            )
            
            # Add to session metadata for JSON
            file_manager.save_result(
                session_id, 
                upload_id, 
                f'mediapipe_{noise_filter}_json', 
                output_json_path, 
                metadata={
                    'model': 'mediapipe',
                    'filter': noise_filter,
                    'window': filter_window,
                    'type': 'json'
                }
            )
            
            print(f"Processed {model} for job {job_id}, saved video to {output_video_path} and JSON to {output_json_path}")
        else:
            error_msg = f"Failed to process video with {model} model"
            print(error_msg)
            jobs[job_id]['errors'].append(error_msg)
    # elif model == 'fourdhumans':
    #     # Create output paths directly in the upload directory
    #     output_video_path = os.path.join(upload_dir, f'fourdhumans_{noise_filter}.mp4')
    #     output_json_path = os.path.join(upload_dir, f'fourdhumans_{noise_filter}.json')
        
    #     # Initialize model
    #     from models.fdhumans_model import FourDHumanWrapper
    #     fdh_model = FourDHumanWrapper()
        
    #     # Process the video
    #     success = fdh_model.process_video(
    #         input_path, 
    #         output_video_path, 
    #         method=noise_filter,
    #         filter_window=filter_window,
    #         output_json_path=output_json_path
    #     )
        
    #     if success:
    #         # Add to job results
    #         result_files = {
    #             'video': output_video_path
    #         }
            
    #         # Add JSON if it exists
    #         if os.path.exists(output_json_path):
    #             result_files['json'] = output_json_path
            
    #         jobs[job_id]['results'][model] = result_files
            
    #         # Add to session metadata for video
    #         file_manager.save_result(
    #             session_id, 
    #             upload_id, 
    #             f'fourdhumans_{noise_filter}', 
    #             output_video_path, 
    #             metadata={
    #                 'model': 'fourdhumans',
    #                 'filter': noise_filter,
    #                 'window': filter_window,
    #                 'type': 'video'
    #             }
    #         )
            
    #         # Add to session metadata for JSON if it exists
    #         if os.path.exists(output_json_path):
    #             file_manager.save_result(
    #                 session_id, 
    #                 upload_id, 
    #                 f'fourdhumans_{noise_filter}_json', 
    #                 output_json_path, 
    #                 metadata={
    #                     'model': 'fourdhumans',
    #                     'filter': noise_filter,
    #                     'window': filter_window,
    #                     'type': 'json'
    #                 }
    #             )
            
    #         print(f"Processed {model} for job {job_id}, saved to {output_video_path}")
    #     else:
    #         error_msg = f"Failed to process video with {model} model"
    #         print(error_msg)
    #         jobs[job_id]['errors'].append(error_msg)
    elif model.startswith('sapiens_'):
        # Create output paths directly in the upload directory
        output_video_path = os.path.join(upload_dir, f'{model}_{noise_filter}.mp4')
        output_json_path = os.path.join(upload_dir, f'{model}_{noise_filter}.json')
        
        # Get model path
        model_path = app.config['SAPIENS_MODELS'].get(model)
        # Get batch size and max frames from config
        batch_size = app.config['MODEL_BATCH_SIZES'].get(model, app.config.get('BATCH_SIZE', 4))
        max_frames = app.config['MAX_FRAMES'].get(model, app.config.get('MAX_FRAMES', 600))
        if not model_path:
            error_msg = f"Model path not found for {model}"
            print(error_msg)
            jobs[job_id]['errors'].append(error_msg)
            return
        
        # Initialize model with explicit CPU fallback if CUDA runs out of memory
        try:
            from models.sapiens import SapiensProcessor
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            try:
                sapiens_processor = SapiensProcessor(model_path, session_id=session_id, upload_id=upload_id,device=device, batch_size=batch_size, save_img_flag=False)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA out of memory for {model}, falling back to CPU")
                    sapiens_processor = SapiensProcessor(model_path, device="cpu", batch_size=batch_size, save_img_flag=False)
                else:
                    raise
            
            # Process the video
            success = sapiens_processor.process_video(
                input_path, 
                output_video_path, 
                method=noise_filter,
                filter_window=filter_window,
                output_json_path=output_json_path,
                save_keypoints=True,
                sample_rate=sample_rate,
                max_frames=max_frames
            )
            
            # Explicitly delete the processor to release CUDA memory
            del sapiens_processor
            
            if success:
                # Add to job results
                jobs[job_id]['results'][model] = {
                    'video': output_video_path,
                    'json': output_json_path if os.path.exists(output_json_path) else None
                }
                
                # Add to session metadata for video
                file_manager.save_result(
                    session_id, 
                    upload_id, 
                    f'{model}_{noise_filter}', 
                    output_video_path, 
                    metadata={
                        'model': model,
                        'filter': noise_filter,
                        'window': filter_window,
                        'type': 'video'
                    }
                )
                
                # Add to session metadata for JSON if it exists
                if os.path.exists(output_json_path):
                    file_manager.save_result(
                        session_id, 
                        upload_id, 
                        f'{model}_{noise_filter}_json', 
                        output_json_path, 
                        metadata={
                            'model': model,
                            'filter': noise_filter,
                            'window': filter_window,
                            'type': 'json'
                        }
                    )
                
                print(f"Processed {model} for job {job_id}, saved video to {output_video_path}")
            else:
                error_msg = f"Failed to process video with {model} model"
                print(error_msg)
                jobs[job_id]['errors'].append(error_msg)
                
        except Exception as e:
            print(f"Error processing {model}: {str(e)}")
            jobs[job_id]['errors'].append(f"Error processing {model}: {str(e)}")
            
            # Ensure we clean up even on failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

@process_bp.route('/process', methods=['POST'])
def process_video():
    """Start video processing with selected models and filters"""
    # Get request data
    data = request.json
    
    if not data:
        return jsonify({
            'status': 'error',
            'message': 'No data provided. Please check your request.'
        }), 400
    
    # Get session ID
    if 'session_id' not in session:
        return jsonify({
            'status': 'error',
            'message': 'No active session. Please refresh the page.'
        }), 400
    
    session_id = session['session_id']
    
    # Required parameters
    upload_id = data.get('upload_id')
    selected_models = data.get('models', [])
    
    # Optional parameters with defaults
    noise_filter = data.get('noise_filter', 'original')
    filter_window = int(data.get('filter_window', 5))
    sample_rate = int(data.get('sample_rate', 1))  # Sample rate for keypoint data
    
    if not upload_id:
        return jsonify({
            'status': 'error', 
            'message': 'No upload ID provided. Please upload a video first.'
        }), 400
    
    if not selected_models:
        return jsonify({
            'status': 'error',
            'message': 'No models selected. Please select at least one model.'
        }), 400
    
    # Validate filter method
    if noise_filter not in current_app.config['NOISE_FILTERS']:
        return jsonify({
            'status': 'error',
            'message': f"Invalid noise filter: {noise_filter}. Available filters: {', '.join(current_app.config['NOISE_FILTERS'])}"
        }), 400
    
    # Validate selected models
    valid_models = ['mediapipe', 'fourdhumans'] + list(current_app.config['SAPIENS_MODELS'].keys())
    for model in selected_models:
        if model not in valid_models:
            return jsonify({
                'status': 'error',
                'message': f"Invalid model: {model}. Available models: {', '.join(valid_models)}"
            }), 400
    
    # Initialize file manager
    file_manager = FileManager(current_app.config['STORAGE_BASE_DIR'])
    
    # Check if session exists
    session_data, error = file_manager.get_session_details(session_id)
    if not session_data:
        return jsonify({
            'status': 'error',
            'message': error or 'Session not found'
        }), 404
    
    # Get upload file path
    input_path, error = file_manager.get_upload_path(session_id, upload_id)
    if not input_path:
        return jsonify({
            'status': 'error',
            'message': error or 'Upload not found'
        }), 404
    
    # Check if file exists
    if not os.path.exists(input_path):
        return jsonify({
            'status': 'error',
            'message': 'The uploaded file could not be found. Please upload again.'
        }), 404
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Create job entry
    jobs[job_id] = {
        'id': job_id,
        'session_id': session_id,
        'upload_id': upload_id,
        'status': 'queued',
        'models': selected_models,
        'noise_filter': noise_filter,
        'filter_window': filter_window,
        'sample_rate': sample_rate,
        'results': {},
        'errors': []
    }
    # get current app
    app = current_app._get_current_object()
    # Start processing in background
    thread = threading.Thread(
        target=process_video_task,
        args=(app, job_id, session_id, upload_id, input_path, selected_models, noise_filter, filter_window, sample_rate)
    )
    thread.daemon = True
    thread.start()
    
    print(f"Started processing job {job_id} for upload {upload_id}")
    
    return jsonify({
        'status': 'success',
        'message': 'Processing started',
        'job_id': job_id,
        'session_id': session_id,
        'upload_id': upload_id,
        'models': selected_models,
        'noise_filter': noise_filter,
        'filter_window': filter_window,
        'sample_rate': sample_rate
    }), 202

@process_bp.route('/available-models', methods=['GET'])
def get_available_models():
    """Get available models and filters"""
    models = {
        'pose_models': [
            {
                'id': 'mediapipe',
                'name': 'MediaPipe',
                'description': 'Google\'s MediaPipe Pose model providing 33 landmarks',
                'output_formats': ['video', 'json']
            },
            {
                'id': 'fourdhumans',
                'name': '4DHumans',
                'description': '3D mesh reconstruction model for human pose and shape',
                'output_formats': ['video']
            }
        ],
        'sapiens_models': [
            {
                'id': model_id,
                'name': f'Sapiens {model_id.split("_")[1]}',
                'description': f'High-accuracy Sapiens model with {model_id.split("_")[1]} parameters',
                'output_formats': ['video', 'json']
            }
            for model_id in current_app.config['SAPIENS_MODELS'].keys()
        ],
        'filters': [
            {
                'id': filter_id,
                'name': filter_id.capitalize(),
                'description': f'{filter_id.capitalize()} noise filter'
            }
            for filter_id in current_app.config['NOISE_FILTERS']
        ],
        'interpolation_methods': [
            {
                'id': method_id,
                'name': method_id.capitalize() if method_id != 'no interpolation' else 'No Interpolation',
                'description': (
                    f'{method_id.capitalize()} interpolation' 
                    if method_id not in ['no interpolation', 'kalman', 'wiener'] 
                    else f'{method_id.capitalize()} filter' if method_id != 'no interpolation' 
                    else 'Original without interpolation'
                )
            }
            for method_id in current_app.config['INTERPOLATION_METHODS']
        ]
    }
    
    return jsonify({
        'status': 'success',
        'models': models
    })