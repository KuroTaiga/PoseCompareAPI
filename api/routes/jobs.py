from flask import Blueprint, jsonify, request, url_for, current_app
import logging
import os

# Import the jobs dictionary from process.py
from routes.process import jobs

logger = logging.getLogger(__name__)

# Create blueprint
jobs_bp = Blueprint('jobs', __name__, url_prefix='/api/jobs')

@jobs_bp.route('/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get the status of a specific job"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    # Create response with job details
    response = {
        'id': job['id'],
        'status': job['status'],
        'models': job['models'],
        'filename': job['filename'],
        'noise_filter': job['noise_filter'],
        'filter_window': job['filter_window']
    }
    
    # If job completed or failed, include additional information
    if job['status'] == 'completed':
        result_urls = {}
        for model, path in job['results'].items():
            # Convert file path to URL
            relative_path = path.replace('static/', '')
            result_urls[model] = url_for('static', filename=relative_path, _external=True)
        
        response['results'] = result_urls
    
    if job['errors']:
        response['errors'] = job['errors']
    
    return jsonify(response)

@jobs_bp.route('/<job_id>/result', methods=['GET'])
def get_job_result(job_id):
    """Get the results of a completed job"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({
            'status': job['status'],
            'message': 'Job not completed yet'
        }), 202
    
    # Return results with URLs to the processed videos
    result_urls = {}
    for model, path in job['results'].items():
        # Convert file path to URL
        relative_path = path.replace('static/', '')
        result_urls[model] = url_for('static', filename=relative_path, _external=True)
    
    return jsonify({
        'status': 'success',
        'job_id': job_id,
        'results': result_urls
    })

@jobs_bp.route('/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Delete a job and its associated files"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    # Delete result files
    for _, path in job['results'].items():
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Deleted result file: {path}")
        except Exception as e:
            logger.error(f"Error deleting result file {path}: {str(e)}")
    
    # Delete uploaded file if it exists
    try:
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], job['filename'])
        if os.path.exists(upload_path):
            os.remove(upload_path)
            logger.info(f"Deleted upload file: {upload_path}")
    except Exception as e:
        logger.error(f"Error deleting upload file: {str(e)}")
    
    # Remove job from memory
    del jobs[job_id]
    
    return jsonify({
        'status': 'success',
        'message': 'Job and associated files deleted'
    })