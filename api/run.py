#!/usr/bin/env python
import os
import sys
from app import create_app

if __name__ == '__main__':
    # Get environment from environment variable or default to development
    env = os.environ.get('FLASK_ENV', 'development')
    
    # Make sure it's a valid environment
    valid_envs = ['development', 'production', 'testing']
    if env not in valid_envs:
        print(f"Error: '{env}' is not a valid environment. Choose from: {', '.join(valid_envs)}")
        sys.exit(1)
    
    # Create app with proper configuration
    app = create_app(env)
    
    # Get host and port from environment variables or use defaults
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    print(f"Starting Pose Processing API in {env.upper()} mode")
    print(f"Server running at http://{host}:{port}")
    
    # Run the application
    app.run(host=host, port=port, debug=(env == 'development'))