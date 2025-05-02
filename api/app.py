from flask import Flask, jsonify, session, send_from_directory
from flask_cors import CORS
from flask_session import Session
import os
import threading
import time
from utils.file_manager import FileManager
from config import config_by_name

# Import routes
from routes.upload import upload_bp
from routes.process import process_bp
from routes.jobs import jobs_bp
# from routes.heatmap import heatmap_bp  # Skipped per request

# Background thread for session cleanup
def cleanup_task(app, interval_hours=1):
    """Background thread to clean up expired sessions"""
    with app.app_context():
        while True:
            try:
                # Sleep first to avoid immediate cleanup on startup
                time.sleep(interval_hours * 3600)  # Convert hours to seconds
                
                # Get max age from config
                max_age_hours = app.config.get('PERMANENT_SESSION_LIFETIME', 5 * 3600) / 3600
                
                # Clean up expired sessions
                file_manager = FileManager(app.config['STORAGE_BASE_DIR'])
                deleted_count = file_manager.cleanup_expired_sessions(max_age_hours)
                
                print(f"Cleaned up {deleted_count} expired sessions (older than {max_age_hours:.1f} hours)")
            except Exception as e:
                print(f"Error in cleanup task: {str(e)}")
def jobs_cleanup_task(app, interval_hours=2):
    """Background thread to clean up old jobs"""
    from routes.jobs import cleanup_old_jobs
    
    with app.app_context():
        while True:
            try:
                # Sleep first to avoid immediate cleanup on startup
                time.sleep(interval_hours * 3600)  # Convert hours to seconds
                
                # Clean up old jobs
                deleted_count = cleanup_old_jobs()
                
                print(f"Cleaned up {deleted_count} old jobs")
            except Exception as e:
                print(f"Error in jobs cleanup task: {str(e)}")

def create_app(config_name='development'):
    """Create and configure the Flask application"""
    print(f"Creating app with configuration: {config_name}")
    
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    
    # Load the appropriate configuration
    app.config.from_object(config_by_name[config_name])
    
    # Verify config is loaded
    print(f"STORAGE_BASE_DIR set to: {app.config.get('STORAGE_BASE_DIR', 'Not found')}")
    
    # Enable CORS with credentials support
    CORS(app, supports_credentials=True)
    
    # Make sure STORAGE_BASE_DIR exists in config
    if 'STORAGE_BASE_DIR' not in app.config:
        print("Warning: STORAGE_BASE_DIR not found in config, using default 'storage'")
        app.config['STORAGE_BASE_DIR'] = 'storage'
    
    # Configure Flask-Session
    app.config['SESSION_FILE_DIR'] = os.path.join(app.config['STORAGE_BASE_DIR'], 'flask_sessions')
    Session(app)
    
    # Ensure necessary directories exist
    try:
        os.makedirs(app.instance_path, exist_ok=True)
        
        # Create storage structure
        storage_base = app.config['STORAGE_BASE_DIR']
        os.makedirs(os.path.join(storage_base, 'sessions'), exist_ok=True)
        
        # Create Flask session directory
        os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
        
        print("Created necessary directories for application")
    except OSError as e:
        print(f"Error creating directories: {e}")
    
    # Start session cleanup thread (only in production)
    if config_name == 'production':
        cleanup_thread = threading.Thread(
            target=cleanup_task,
            args=(app, 1),  # Run every 1 hour
            daemon=True
        )
        cleanup_thread.start()
        print("Started background session cleanup thread")
        # Jobs cleanup thread
        jobs_cleanup_thread = threading.Thread(
            target=jobs_cleanup_task,
            args=(app, 2),  # Run every 2 hours
            daemon=True
        )
        jobs_cleanup_thread.start()
        
        print("Started background cleanup threads")
    
    # Register blueprints
    app.register_blueprint(upload_bp)
    app.register_blueprint(process_bp)
    app.register_blueprint(jobs_bp)
    # app.register_blueprint(heatmap_bp)  # Skipped per request
    
    # Route to serve static index page
    @app.route('/')
    def index():
        return send_from_directory('static', 'index.html')
    
    # Route to serve files from storage/sessions directory
    @app.route('/session-files/<session_id>/<path:filename>')
    def serve_session_file(session_id, filename):
        # Check if the session ID matches the current session
        if 'session_id' in session and session_id != session['session_id']:
            return jsonify({
                'status': 'error',
                'message': 'You do not have permission to access this file'
            }), 403
        
        # Construct the full path within the sessions directory
        session_dir = os.path.join(app.config['STORAGE_BASE_DIR'], 'sessions', session_id)
        
        # Security check to prevent directory traversal
        requested_path = os.path.abspath(os.path.join(session_dir, filename))
        if not requested_path.startswith(os.path.abspath(session_dir)):
            return jsonify({
                'status': 'error',
                'message': 'Invalid file path'
            }), 400
        
        # Get the directory part of the path
        dir_path = os.path.dirname(requested_path)
        file_name = os.path.basename(requested_path)
        
        return send_from_directory(dir_path, file_name)
    
    # API information endpoint
    @app.route('/api/info')
    def api_info():
        # Get session ID if available
        session_id = session.get('session_id', None)
        
        response = {
            'status': 'ok',
            'message': 'Pose Estimation API is running',
            'version': '1.0.0',
            'endpoints': {
                'session': '/api/session',
                'upload_video': '/api/upload',
                'get_files': '/api/files',
                'process_video': '/api/process',
                'available_models': '/api/process/available-models',
                'job_status': '/api/jobs/<job_id>',
                'job_result': '/api/jobs/<job_id>/result',
                'session_files': '/session-files/<session_id>/<filename>'
            }
        }
        
        # Add session info if available
        if session_id:
            # Get session details from file manager
            file_manager = FileManager(app.config['STORAGE_BASE_DIR'])
            session_details, error = file_manager.get_session_details(session_id)
            
            if session_details:
                response['session'] = {
                    'id': session_id,
                    'upload_count': len(session_details.get('uploads', [])),
                    'result_count': len(session_details.get('results', []))
                }
        
        return jsonify(response)
    
    return app

if __name__ == '__main__':
    app = create_app('development')
    # app.run(debug=True)
    app.run(debug=False,use_reloader=False)