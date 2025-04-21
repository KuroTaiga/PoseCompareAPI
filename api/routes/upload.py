import os
import uuid
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import logging

logger = logging.getLogger(__name__)

# Create blueprint
upload_bp = Blueprint('upload', __name__, url_prefix='/api')

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle video file uploads"""
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    
    # If user does not select file, browser might submit an empty file
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        # Generate a unique filename
        job_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        basename, extension = os.path.splitext(filename)
        unique_filename = f"{basename}_{job_id}{extension}"
        
        # Save the file
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        logger.info(f"File saved: {file_path}")
        
        return jsonify({
            'status': 'success',
            'message': 'File uploaded successfully',
            'job_id': job_id,
            'filename': unique_filename,
            'original_filename': filename
        }), 201
    
    return jsonify({'error': 'File type not allowed'}), 400