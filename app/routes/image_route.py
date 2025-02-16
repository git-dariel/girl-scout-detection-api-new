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
from functools import lru_cache
import gc

image_route = Blueprint("image_route", __name__)

# Initialize TensorBoard logger
tensorboard_logger = TensorBoardLogger()

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
    try:
        start_time = time.time()
        print("Starting image processing...")
        
        # Load image in chunks to reduce memory usage
        image, error = handle_file_upload()
        if error:
            return jsonify({"error": error}), 400
        
        # Process image in smaller batches
        result = ImageProcessingService.process_image(image)
        print("Image processing completed successfully")
        
        # Free up memory after processing
        del image
        gc.collect()
        
        # Resize and optimize image
        image = resize_image_if_needed(image)
        image = optimize_image_for_processing(image)
        
        # Encode original image with compression
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        _, buffer = cv2.imencode(".png", image, encode_params)
        original_image_bytes = buffer.tobytes()
        
        try:
            # Process image and handle uploads
            result, original_image_url, processed_image_url = process_image_with_executor(
                image, original_image_bytes
            )
        except TimeoutError as e:
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        # Log the request to TensorBoard and get graph URL
        processing_time = time.time() - start_time
        image_size = image.size if hasattr(image, 'size') else len(original_image_bytes)
        graph_url, graph_explanations = tensorboard_logger.log_request(
            endpoint="detect_uniform",
            processing_time=processing_time,
            image_size=image_size,
            result=result
        )
        
        # Prepare response data
        response_data = {
            "is_authentic": result["is_authentic"],
            "confidence_score": result["confidence_score"],
            "message": result["message"],
            "details": result["details"],
            "original_image_url": original_image_url,
            "processed_image_url": processed_image_url,
        }
        
        if graph_url:
            response_data["graph_url"] = graph_url
            
        if graph_explanations:
            response_data["graph_analysis"] = graph_explanations

        # Save detection result to database
        try:
            # Add raw predictions and uniform type to the data
            detection_data = {
                **response_data,
                "raw_predictions": result.get("raw_predictions", []),
                "uniform_type": result.get("uniform_type", "unknown")
            }
            
            # Create record in database
            saved_uniform = DetectedUniformService.create_detected_uniform(detection_data)
            
            # Add the database ID to the response
            response_data["detection_id"] = str(saved_uniform._id)
            
        except Exception as e:
            print(f"Warning: Failed to save detection result to database: {str(e)}")
            # Continue with the response even if database save fails
            
        return jsonify(response_data)
        
    except Exception as e:
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