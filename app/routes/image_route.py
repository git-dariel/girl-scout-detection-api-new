from flask import Blueprint, jsonify, current_app
from flasgger import swag_from
from app.middlewares.upload_middleware import handle_file_upload
from app.utils.tensorboard_logger import TensorBoardLogger
from app.utils.image_processing_utils import (
    resize_image_if_needed,
    optimize_image_for_processing,
    process_image_with_executor
)
from app.services.detected_uniform_service import DetectedUniformService
from app.services.image_processing_service import ImageProcessingService
import time
import os
import cv2
import numpy as np
from functools import lru_cache
import gc

image_route = Blueprint("image_route", __name__)

# Initialize TensorBoard logger
tensorboard_logger = TensorBoardLogger()

def cleanup_resources(*objects):
    """Helper function to clean up memory"""
    for obj in objects:
        if obj is not None:
            del obj
    gc.collect()

@image_route.route("/api/detect-uniform", methods=["POST"])
@swag_from({
    "parameters": [
        {
            "name": "file",
            "in": "formData",
            "type": "file",
            "required": True,
            "description": "The image file containing the Girl Scout uniform to analyze"
        }
    ],
    "responses": {
        200: {
            "description": "Girl Scout uniform detection results"
        },
        400: {
            "description": "Invalid input"
        },
        500: {
            "description": "Internal server error"
        }
    }
})
def detect_uniform():
    image = None
    buffer = None
    try:
        start_time = time.time()
        print("Starting image processing...")
        
        # Load image in chunks to reduce memory usage
        image, error = handle_file_upload()
        if error:
            return jsonify({"error": error}), 400
        
        # Aggressively resize image to reduce memory usage
        max_dimension = 800  # Limit maximum dimension
        height, width = image.shape[:2]
        if width > max_dimension or height > max_dimension:
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to RGB and optimize memory usage
        if image.shape[2] == 4:  # If RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Further optimize image
        image = optimize_image_for_processing(image)
        
        # Process image with lower memory usage
        with current_app.app_context():
            try:
                # Encode with lower quality for memory savings
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]  # Use JPEG instead of PNG
                _, buffer = cv2.imencode(".jpg", image, encode_params)
                original_image_bytes = buffer.tobytes()
                
                # Process image and handle uploads
                result, original_image_url, processed_image_url = process_image_with_executor(
                    image.copy(), original_image_bytes
                )
                
                # Clean up large objects immediately
                cleanup_resources(buffer, original_image_bytes)
                
            except TimeoutError as e:
                cleanup_resources(image, buffer)
                return jsonify({"error": str(e)}), 500
            except Exception as e:
                cleanup_resources(image, buffer)
                return jsonify({"error": f"Failed to process image: {str(e)}"}), 500
        
        # Log the request to TensorBoard and get graph URL
        processing_time = time.time() - start_time
        image_size = len(original_image_bytes) if 'original_image_bytes' in locals() else 0
        
        graph_url, graph_explanations = tensorboard_logger.log_request(
            endpoint="detect_uniform",
            processing_time=processing_time,
            image_size=image_size,
            result=result
        )
        
        # Prepare complete response data
        response_data = {
            "is_authentic": result["is_authentic"],
            "confidence_score": result["confidence_score"],
            "message": result["message"],
            "details": result["details"],
            "original_image_url": original_image_url,
            "processed_image_url": processed_image_url,
        }
        
        # Add graph data if available
        if graph_url:
            response_data["graph_url"] = graph_url
        if graph_explanations:
            response_data["graph_analysis"] = graph_explanations
        
        # Save detection result to database
        try:
            detection_data = {
                **response_data,
                "raw_predictions": result.get("raw_predictions", []),
                "uniform_type": result.get("uniform_type", "unknown")
            }
            saved_uniform = DetectedUniformService.create_detected_uniform(detection_data)
            response_data["detection_id"] = str(saved_uniform._id)
        except Exception as e:
            print(f"Warning: Failed to save detection result to database: {str(e)}")
        
        # Final cleanup
        cleanup_resources(image, buffer)
        
        return jsonify(response_data)
        
    except Exception as e:
        cleanup_resources(image, buffer)
        print(f"Error in detect_uniform: {str(e)}")
        current_app.logger.error(f"Unexpected error in detect_uniform: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

@image_route.route("/api/get/processed-image/<image_id>", methods=["GET"])
@swag_from({
     "parameters": [
        {
            "name": "image_id",
            "in": "path",
            "type": "string",
            "required": True,
            "description": "The ID of the processed image"
        }
    ],
    "responses": {
        200: {
            "description": "URL of the processed image"
        },
        500: {
            "description": "Internal server error"
        }
    }
})
@lru_cache(maxsize=100)  # Cache up to 100 image URLs
def get_processed_image(image_id):
    try:
        image_url = f"https://res.cloudinary.com/{os.environ.get('CLOUDINARY_CLOUD_NAME')}/image/upload/processed_uniforms/{image_id}"
        return jsonify({"image_url": image_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500