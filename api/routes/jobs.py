from flask import Blueprint, jsonify, request, url_for, current_app, session
import os
from utils.file_manager import FileManager
from routes.process import jobs

# Create blueprint
jobs_bp = Blueprint('jobs', __name__, url_prefix='/api/jobs')

@jobs_bp.route('/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get the status of a specific job"""
    if job_id not in jobs:
        return jsonify({
            'status': 'error',
            'message': 'Job not found. It may have expired or been deleted.'
        }), 404
    
    job = jobs[job_id]
    
    # Check if the job belongs to the current session
    if 'session_id' in session and job.get('session_id') != session['session_id']:
        return jsonify({
            'status': 'error',
            'message': 'You do not have permission to access this job.'
        }), 403
    
    # Create response with job details
    response = {
        'status': 'success',
        'job': {
            'id': job['id'],
            'status': job['status'],
            'models': job['models'],
            'noise_filter': job['noise_filter'],
            'filter_window': job['filter_window'],
            'sample_rate': job.get('sample_rate', 1),
            'session_id': job.get('session_id'),
            'upload_id': job.get('upload_id')
        }
    }
    
    # If job completed or failed, include additional information
    if job['status'] == 'completed':
        # If we have session ID and upload ID, create URLs for the results
        if 'session_id' in job and 'upload_id' in job:
            session_id = job['session_id']
            
            # Format results with videos and JSONs
            formatted_results = {}
            
            for model, result_data in job['results'].items():
                model_results = {}
                
                # Video result
                if 'video' in result_data and os.path.exists(result_data['video']):
                    # Get relative path from session directory
                    rel_path = os.path.relpath(
                        result_data['video'], 
                        os.path.join(current_app.config['STORAGE_BASE_DIR'], 'sessions', session_id)
                    )
                    
                    # Create URL
                    model_results['video'] = url_for(
                        'serve_session_file',
                        session_id=session_id,
                        filename=rel_path,
                        _external=True
                    )
                
                # JSON result
                if 'json' in result_data and result_data['json'] and os.path.exists(result_data['json']):
                    # Get relative path from session directory
                    rel_path = os.path.relpath(
                        result_data['json'], 
                        os.path.join(current_app.config['STORAGE_BASE_DIR'], 'sessions', session_id)
                    )
                    
                    # Create URL
                    model_results['json'] = url_for(
                        'serve_session_file',
                        session_id=session_id,
                        filename=rel_path,
                        _external=True
                    )
                
                if model_results:
                    formatted_results[model] = model_results
            
            response['results'] = formatted_results
    
    if job['errors']:
        response['errors'] = job['errors']
    
    return jsonify(response)

@jobs_bp.route('/<job_id>/result', methods=['GET'])
def get_job_result(job_id):
    """Get the results of a completed job"""
    if job_id not in jobs:
        return jsonify({
            'status': 'error',
            'message': 'Job not found. It may have expired or been deleted.'
        }), 404
    
    job = jobs[job_id]
    
    # Check if the job belongs to the current session
    if 'session_id' in session and job.get('session_id') != session['session_id']:
        return jsonify({
            'status': 'error',
            'message': 'You do not have permission to access this job.'
        }), 403
    
    if job['status'] != 'completed':
        return jsonify({
            'status': 'pending',
            'message': f'Job not completed yet. Current status: {job["status"]}'
        }), 202
    
    # If we have session ID and upload ID, create URLs for the results
    if 'session_id' in job and 'upload_id' in job:
        session_id = job['session_id']
        
        # Format results with videos and JSONs
        formatted_results = {}
        
        for model, result_data in job['results'].items():
            model_results = {}
            
            # Video result
            if 'video' in result_data and os.path.exists(result_data['video']):
                # Get relative path from session directory
                rel_path = os.path.relpath(
                    result_data['video'], 
                    os.path.join(current_app.config['STORAGE_BASE_DIR'], 'sessions', session_id)
                )
                
                # Create URL
                model_results['video'] = url_for(
                    'serve_session_file',
                    session_id=session_id,
                    filename=rel_path,
                    _external=True
                )
            
            # JSON result
            if 'json' in result_data and result_data['json'] and os.path.exists(result_data['json']):
                # Get relative path from session directory
                rel_path = os.path.relpath(
                    result_data['json'], 
                    os.path.join(current_app.config['STORAGE_BASE_DIR'], 'sessions', session_id)
                )
                
                # Create URL
                model_results['json'] = url_for(
                    'serve_session_file',
                    session_id=session_id,
                    filename=rel_path,
                    _external=True
                )
            
            if model_results:
                formatted_results[model] = model_results
        
        return jsonify({
            'status': 'success',
            'job_id': job_id,
            'results': formatted_results
        })
    
    # No session/upload info, return error
    return jsonify({
        'status': 'error',
        'message': 'Job results unavailable'
    }), 500

@jobs_bp.route('/<job_id>/download', methods=['GET'])
def download_job_results(job_id):
    """Get all results for a job as a downloadable package"""
    if job_id not in jobs:
        return jsonify({
            'status': 'error',
            'message': 'Job not found. It may have expired or been deleted.'
        }), 404
    
    job = jobs[job_id]
    
    # Check if the job belongs to the current session
    if 'session_id' in session and job.get('session_id') != session['session_id']:
        return jsonify({
            'status': 'error',
            'message': 'You do not have permission to access this job.'
        }), 403
    
    if job['status'] != 'completed':
        return jsonify({
            'status': 'pending',
            'message': f'Job not completed yet. Current status: {job["status"]}'
        }), 202
    
    # If we have session ID and upload ID, use file manager to get all files
    if 'session_id' in job and 'upload_id' in job:
        session_id = job['session_id']
        upload_id = job['upload_id']
        
        # Initialize file manager
        file_manager = FileManager(current_app.config['STORAGE_BASE_DIR'])
        
        # Get all files for this upload
        files, error = file_manager.get_all_files_for_upload(session_id, upload_id)
        
        if not files:
            return jsonify({
                'status': 'error',
                'message': error or 'No files found for this job'
            }), 404
        
        # Create URLs for all files
        file_urls = {}
        for file_type, file_path in files.items():
            # Get relative path from session directory
            rel_path = os.path.relpath(
                file_path, 
                os.path.join(current_app.config['STORAGE_BASE_DIR'], 'sessions', session_id)
            )
            
            # Determine file format (video or json)
            file_format = 'video' if file_path.endswith('.mp4') else 'json' if file_path.endswith('.json') else 'other'
            
            # Get the model name from the filename
            filename = os.path.basename(file_path)
            model_name = filename.split('_')[0] if '_' in filename else 'unknown'
            
            # Group files by model
            if model_name not in file_urls:
                file_urls[model_name] = {}
                
            # Add URL
            file_urls[model_name][file_format] = url_for(
                'serve_session_file',
                session_id=session_id,
                filename=rel_path,
                _external=True
            )
        
        return jsonify({
            'status': 'success',
            'job_id': job_id,
            'session_id': session_id,
            'upload_id': upload_id,
            'files': file_urls
        })
    
    # No session/upload info, return error
    return jsonify({
        'status': 'error',
        'message': 'Job results unavailable'
    }), 500

@jobs_bp.route('/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Delete a job (note: this doesn't delete the files, just the job record)"""
    if job_id not in jobs:
        return jsonify({
            'status': 'error',
            'message': 'Job not found. It may have already been deleted.'
        }), 404
    
    job = jobs[job_id]
    
    # Check if the job belongs to the current session
    if 'session_id' in session and job.get('session_id') != session['session_id']:
        return jsonify({
            'status': 'error',
            'message': 'You do not have permission to delete this job.'
        }), 403
    
    # Remove job from memory
    del jobs[job_id]
    
    return jsonify({
        'status': 'success',
        'message': 'Job deleted successfully'
    })