from flask import Flask, jsonify, session, send_from_directory
from flask_cors import CORS
from flask_session import Session
import os
from utils.file_manager import FileManager
from config import config_by_name

# Import routes
from routes.upload import upload_bp
# from routes.process import process_bp
# from routes.jobs import jobs_bp
# from routes.heatmap import heatmap_bp  # Skipped per request

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
        os.makedirs(os.path.join(storage_base, 'uploads'), exist_ok=True)
        os.makedirs(os.path.join(storage_base, 'results'), exist_ok=True)
        os.makedirs(os.path.join(storage_base, 'temp'), exist_ok=True)
        
        # Create Flask session directory
        os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
        
        print("Created necessary directories for application")
    except OSError as e:
        print(f"Error creating directories: {e}")
    
    # Register blueprints
    app.register_blueprint(upload_bp)
    # app.register_blueprint(process_bp)
    # app.register_blueprint(jobs_bp)
    # app.register_blueprint(heatmap_bp)  # Skipped per request
    
    # Route to serve static index page
    @app.route('/')
    def index():
        return send_from_directory('static', 'index.html')
    
    # Route to serve files from storage
    @app.route('/storage/<path:filename>')
    def serve_storage_file(filename):
        # Check if the session ID in the path matches the current session
        path_parts = filename.split('/')
        if len(path_parts) >= 2:
            path_session_id = path_parts[0]
            if 'session_id' in session and path_session_id != session['session_id']:
                return jsonify({
                    'status': 'error',
                    'message': 'You do not have permission to access this file'
                }), 403
        
        return send_from_directory(app.config['STORAGE_BASE_DIR'], filename)
    
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
    app.run(debug=True)