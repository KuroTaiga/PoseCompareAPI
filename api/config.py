import os

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_key_replace_in_production')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max upload size

    SESSION_COOKIE_NAME = 'pose_api_session'
    SESSION_TYPE = 'filesystem'
    SESSION_PERMANENT = True  # Maintain session across requests
    PERMANENT_SESSION_LIFETIME = 3600*5  # 5 hours in seconds

    # File paths
    STORAGE_BASE_DIR = 'storage'
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm'}

    # Model paths
    SAPIENS_CHECKPOINT_ROOT = "api/models/sapiens/checkpoints"
    DEFAULT_CHECKPOINT = SAPIENS_CHECKPOINT_ROOT+"/sapiens_1b_coco_best_coco_AP_821_torchscript.pt2"
    # DEFAULT_CHECKPOINT = SAPIENS_CHECKPOINT_ROOT+"/sapiens_1b_goliath_best_goliath_AP_639.pth"
    SAPIENS_MODELS = {
    'sapiens_2b': SAPIENS_CHECKPOINT_ROOT+'/sapiens_2b_coco_best_coco_AP_822_torchscript.pt2',
    'sapiens_1b': SAPIENS_CHECKPOINT_ROOT+'/sapiens_1b_coco_best_coco_AP_821_torchscript.pt2',
    'sapiens_0.6b': SAPIENS_CHECKPOINT_ROOT+'/sapiens_0.6b_coco_best_coco_AP_812_torchscript.pt2',
    'sapiens_0.3b': SAPIENS_CHECKPOINT_ROOT+'/sapiens_0.3b_coco_best_coco_AP_796_torchscript.pt2'
    # 'sapiens_1b': SAPIENS_CHECKPOINT_ROOT+'/sapiens_1b_goliath_best_goliath_AP_639.pth',
    # 'sapiens_0.6b': SAPIENS_CHECKPOINT_ROOT+'/sapiens_0.6b_goliath_best_goliath_AP_609.pth',
    # 'sapiens_0.3b': SAPIENS_CHECKPOINT_ROOT+'/sapiens_0.3b_goliath_best_goliath_AP_573.pth'
    }

    # Processing parameters
    BATCH_SIZE = 8
    MAX_FRAMES = 1000  # Limit processing to 1000 frames 
    VIDEO_RESIZE_DIMENSIONS = (1024, 768)  # (width, height)

    # API rate limits
    RATE_LIMIT = {
        'default': '100 per day',
        'upload': '10 per minute'
    }

    # Job settings
    JOB_TIMEOUT = 3600  # 1 hour in seconds
    
    # Available filters and methods
    NOISE_FILTERS = [
        'original',
        'butterworth',
        'chebyshev',
        'bessel'
    ]

    INTERPOLATION_METHODS = [
        'no interpolation',
        'kalman',
        'wiener',
        'linear',
        'bilinear',
        'spline',
        'kriging'
    ]

    HEATMAP_TYPES = {
        'motion': 'Visualizes movement intensity across the video',
        'density': 'Shows areas with highest concentration of pose landmarks',
        'trajectory': 'Tracks the path of key body points through the video'
    }
    # Model batch sizes
    MODEL_BATCH_SIZES = {
        'mediapipe': 8,
        'sapiens_0.3b': 8,
        'sapiens_0.6b': 4,
        'sapiens_1b': 2,
        'sapiens_2b': 1  # Smallest batch size for largest model
    }
    
    # Max frames to process in a single video
    MAX_FRAMES = {
        'mediapipe': 1000,
        'sapiens_0.3b': 1000,
        'sapiens_0.6b': 800,
        'sapiens_1b': 600,
        'sapiens_2b': 400  # Fewer frames for largest model
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    DEVELOPMENT = True
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
    # In production, you might want to use different paths
    # Example: STORAGE_BASE_DIR = '/var/www/pose-api/uploads'
    
class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    
# Configuration dictionary
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}