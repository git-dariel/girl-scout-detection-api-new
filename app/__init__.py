from flask import Flask, request
from flask_cors import CORS
from flasgger import Swagger
from app.routes.image_route import image_route
from app.routes.server_route import server_route
from app.config.config import Config
from app.config.cloudinary_config import configure_cloudinary
from app.config.swagger_config import swagger_config
from app.routes.detected_uniform_route import detected_uniform_route
import signal
import os

def create_app():
    app = Flask(__name__)
    
    # Define allowed origins
    allowed_origins = ["https://smart-scouts.vercel.app", "http://localhost:5173", "http://127.0.0.1:5173"]
    
    # Configure CORS - use a simpler approach
    # CORS(app, 
    #      resources={r"/api/*": {"origins": allowed_origins}},
    #      supports_credentials=True,
    #      allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Origin", "X-Requested-With"],
    #      methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    #      expose_headers=["Content-Range", "X-Content-Range"])
    
    # Use a simpler CORS implementation
    app.config.from_object(Config)

    # Add CORS headers to all responses
    @app.after_request
    def after_request(response):
        # Get origin from headers
        origin = request.headers.get('Origin', '')
        
        # For production and development environments
        if origin in allowed_origins or 'smart-scouts.vercel.app' in origin:
            response.headers.add('Access-Control-Allow-Origin', origin)
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
        
        return response
    
    # Handle preflight requests
    @app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
    @app.route('/<path:path>', methods=['OPTIONS'])
    def options_handler(path):
        response = app.make_default_options_response()
        origin = request.headers.get('Origin', '')
        
        if origin in allowed_origins or 'smart-scouts.vercel.app' in origin:
            response.headers.add('Access-Control-Allow-Origin', origin)
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
        
        return response

    # Register blueprints
    app.register_blueprint(image_route)
    app.register_blueprint(server_route)
    app.register_blueprint(detected_uniform_route)

    # Configure Cloudinary
    configure_cloudinary()

    # Initialize Swagger
    Swagger(app, config=swagger_config)

    def handle_shutdown(signum=None, frame=None):
        """Handle server shutdown gracefully and quickly"""
        try:
            print("\nShutting down server...")
            # Force exit after a short delay
            os._exit(0)
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")
            os._exit(1)

    # Register shutdown handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    return app