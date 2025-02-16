import cv2
import numpy as np
import threading
import concurrent.futures
from flask import current_app
from app.services.cloudinary_service import CloudinaryService
from app.services.image_processing_service import ImageProcessingService

# Create a semaphore to limit concurrent processing
processing_semaphore = threading.Semaphore(2)

def resize_image_if_needed(image, max_size=600):
    """Resize image if it's too large while maintaining aspect ratio"""
    height, width = image.shape[:2]
    if height > max_size or width > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

def optimize_image_for_processing(image):
    """Optimize image for faster processing"""
    # Ensure image is in uint8 format for OpenCV operations
    if image.dtype != np.uint8:
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Ensure correct color format
    if len(image.shape) == 2:  # If grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # If RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    return image

def process_image_with_timeout(image):
    """Process image with semaphore to limit concurrent processing"""
    with processing_semaphore:
        return ImageProcessingService.process_image(image)

def process_and_upload_image(image_bytes, is_original=True):
    """Process and upload image to Cloudinary with retry"""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            if is_original:
                return CloudinaryService.upload_original_image(image_bytes)
            return CloudinaryService.upload_processed_image(image_bytes)
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                current_app.logger.error(f"Error uploading image after {max_retries} retries: {str(e)}")
                return None
            current_app.logger.warning(f"Retry {retry_count} for image upload: {str(e)}")

def process_image_with_executor(image, original_image_bytes):
    """Process image and handle uploads with executor"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Start processing the image with higher priority
        future_process = executor.submit(process_image_with_timeout, image)
        
        # Start uploading original image in parallel
        future_original = executor.submit(process_and_upload_image, original_image_bytes, True)
        
        try:
            # Wait for processing to complete with increased timeout
            result = future_process.result(timeout=90)
        except concurrent.futures.TimeoutError:
            executor.shutdown(wait=False)
            raise TimeoutError("Image processing timed out. Please try with a smaller image or try again later.")
        except Exception as e:
            executor.shutdown(wait=False)
            raise Exception(f"Failed to process image: {str(e)}")
        
        # Start uploading processed image
        future_processed = executor.submit(process_and_upload_image, result["processed_image_bytes"], False)
        
        try:
            # Wait for uploads to complete with increased timeout
            original_image_url = future_original.result(timeout=60)
            processed_image_url = future_processed.result(timeout=60)
            
            if not original_image_url or not processed_image_url:
                raise Exception("Failed to upload images")
                
            return result, original_image_url, processed_image_url
            
        except concurrent.futures.TimeoutError:
            raise TimeoutError("Image upload timed out. Please try again later.")
        except Exception as e:
            raise Exception(f"Failed to upload images: {str(e)}") 