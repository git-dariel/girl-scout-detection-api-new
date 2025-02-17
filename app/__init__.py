from flask import Flask
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
    CORS(app)
    app.config.from_object(Config)

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