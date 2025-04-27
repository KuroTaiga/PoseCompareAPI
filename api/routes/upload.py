import os
from flask import Blueprint, request, jsonify, current_app, session, url_for
from werkzeug.utils import secure_filename
from utils.file_manager import FileManager

# Create blueprint
upload_bp = Blueprint('upload', __name__, url_prefix='/api')

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@upload_bp.route('/session', methods=['GET'])
def get_or_create_session():
    """Get the current session ID or create a new one"""
    file_manager = FileManager(current_app.config['STORAGE_BASE_DIR'])
    
    # Check if a session already exists
    if 'session_id' not in session:
        # Create a new session
        session_id = file_manager.create_user_session()
        session['session_id'] = session_id
        print(f"Created new session: {session_id}")
        
        return jsonify({
            'status': 'success',
            'message': 'New session created',
            'session_id': session_id,
            'is_new': True
        })
    
    # Return existing session
    session_id = session['session_id']
    
    # Get session details
    session_details, error = file_manager.get_session_details(session_id)
    
    if session_details is None:
        # Session exists in cookie but not on server, create a new one
        session_id = file_manager.create_user_session()
        session['session_id'] = session_id
        print(f"Recreated session due to error: {error}")
        
        return jsonify({
            'status': 'success',
            'message': 'New session created',
            'error_message': error,
            'session_id': session_id,
            'is_new': True
        })
    
    # Return existing session info
    latest_upload = None
    if session_details['uploads']:
        latest_upload = session_details['uploads'][-1]
        
        # Add file URL to the latest upload
        if 'path' in latest_upload:
            file_path = latest_upload['path']
            file_url = url_for(
                'serve_session_file',
                session_id=session_id,
                filename=os.path.relpath(file_path, os.path.join(current_app.config['STORAGE_BASE_DIR'], 'sessions', session_id)),
                _external=True
            )
            latest_upload['url'] = file_url
    
    return jsonify({
        'status': 'success',
        'message': 'Using existing session',
        'session_id': session_id,
        'is_new': False,
        'upload_count': len(session_details['uploads']),
        'result_count': len(session_details['results']),
        'latest_upload': latest_upload
    })

@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle video file uploads"""
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({
            'status': 'error', 
            'message': 'No file part'
        }), 400
        
    file = request.files['file']
    
    # If user does not select file, browser might submit an empty file
    if file.filename == '':
        return jsonify({
            'status': 'error', 
            'message': 'No selected file'
        }), 400
        
    if file and allowed_file(file.filename):
        # Initialize file manager
        file_manager = FileManager(current_app.config['STORAGE_BASE_DIR'])
        
        # Get or create session
        if 'session_id' not in session:
            session_id = file_manager.create_user_session()
            session['session_id'] = session_id
        else:
            session_id = session['session_id']
        
        # Check if session exists on server
        session_details, error = file_manager.get_session_details(session_id)
        if session_details is None:
            # Create a new session if not found
            session_id = file_manager.create_user_session()
            session['session_id'] = session_id
            print(f"Created new session due to error: {error}")
        
        # Save the upload (this will clear previous uploads and results)
        original_filename = secure_filename(file.filename)
        upload_info, error = file_manager.save_upload(session_id, file, original_filename)
        
        if not upload_info:
            return jsonify({
                'status': 'error',
                'message': error or 'Failed to save upload'
            }), 500
        
        # Create file URL
        file_url = url_for(
            'serve_session_file',
            session_id=session_id,
            filename=os.path.relpath(
                upload_info['path'], 
                os.path.join(current_app.config['STORAGE_BASE_DIR'], 'sessions', session_id)
            ),
            _external=True
        )
        
        print(f"File saved: {upload_info['path']}")
        
        return jsonify({
            'status': 'success',
            'message': 'File uploaded successfully',
            'session_id': session_id,
            'upload_id': upload_info['upload_id'],
            'filename': upload_info['filename'],
            'original_filename': upload_info['original_filename'],
            'path': upload_info['path'],
            'url': file_url,
            'dir': upload_info['dir']
        }), 201
    
    return jsonify({
        'status': 'error', 
        'message': 'File type not allowed. Please upload a video file.'
    }), 400

@upload_bp.route('/files', methods=['GET'])
def get_files():
    """Get files for the current session"""
    # Initialize file manager
    file_manager = FileManager(current_app.config['STORAGE_BASE_DIR'])
    
    # Get session ID
    if 'session_id' not in session:
        return jsonify({
            'status': 'error',
            'message': 'No active session. Please refresh the page.'
        }), 400
    
    session_id = session['session_id']
    
    # Get session details
    session_details, error = file_manager.get_session_details(session_id)
    
    if session_details is None:
        return jsonify({
            'status': 'error',
            'message': error or 'Session not found'
        }), 404
    
    # Add URLs to uploads
    for upload in session_details['uploads']:
        if 'path' in upload:
            file_path = upload['path']
            file_url = url_for(
                'serve_session_file',
                session_id=session_id,
                filename=os.path.relpath(
                    file_path, 
                    os.path.join(current_app.config['STORAGE_BASE_DIR'], 'sessions', session_id)
                ),
                _external=True
            )
            upload['url'] = file_url
    
    # Add URLs to results
    for result in session_details['results']:
        if 'path' in result:
            file_path = result['path']
            file_url = url_for(
                'serve_session_file',
                session_id=session_id,
                filename=os.path.relpath(
                    file_path, 
                    os.path.join(current_app.config['STORAGE_BASE_DIR'], 'sessions', session_id)
                ),
                _external=True
            )
            result['url'] = file_url
    
    return jsonify({
        'status': 'success',
        'session_id': session_id,
        'uploads': session_details['uploads'],
        'results': session_details['results']
    })

@upload_bp.route('/upload/<upload_id>/files', methods=['GET'])
def get_upload_files(upload_id):
    """Get all files for a specific upload"""
    # Initialize file manager
    file_manager = FileManager(current_app.config['STORAGE_BASE_DIR'])
    
    # Get session ID
    if 'session_id' not in session:
        return jsonify({
            'status': 'error',
            'message': 'No active session. Please refresh the page.'
        }), 400
    
    session_id = session['session_id']
    
    # Get all files for this upload
    files, error = file_manager.get_all_files_for_upload(session_id, upload_id)
    
    if not files:
        return jsonify({
            'status': 'error',
            'message': error or 'No files found for this upload'
        }), 404
    
    # Add URLs to the files
    file_urls = {}
    for file_type, file_path in files.items():
        file_url = url_for(
            'serve_session_file',
            session_id=session_id,
            filename=os.path.relpath(
                file_path, 
                os.path.join(current_app.config['STORAGE_BASE_DIR'], 'sessions', session_id)
            ),
            _external=True
        )
        file_urls[file_type] = file_url
    
    return jsonify({
        'status': 'success',
        'session_id': session_id,
        'upload_id': upload_id,
        'files': file_urls
    })