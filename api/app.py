import os
from flask import Flask, jsonify
from flask_cors import CORS
import logging

from config import Config

from routes.upload import upload_bp
# from routes.process import process_bp
# from routes.jobs import jobs_bp
# from routes.heatmap import heatmap_bp


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pose_api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__,instance_relative_config=True)
    CORS(app)
    app.config.from_object(Config)
    try:
        os.makedirs(app.instance_path, exist_ok=True)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
        os.makedirs(app.config['HEATMAPS_FOLDER'], exist_ok=True)
    except OSError:
        logger.error(f"Error creating instance path: {app.instance_path}")
        pass
    # Register blueprints
    app.register_blueprint(upload_bp)
    # app.register_blueprint(process_bp)
    # app.register_blueprint(jobs_bp)
    # app.register_blueprint(heatmap_bp)

    @app.route('/')
    def index():
        return jsonify({
            'status': 'ok',
            'message': 'API up',
            'endpoints': {
                'upload_video': '/api/upload',
                'process_video': '/api/process',
                'job_status': '/api/jobs/<job_id>',
                'job_result': '/api/jobs/<job_id>/result',
                'heatmap_generate': '/api/heatmap/generate',
                'heatmap_types': '/api/heatmap/types'
            }
        })

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)