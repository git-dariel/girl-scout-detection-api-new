import cloudinary.uploader
from flask import current_app
import io

class CloudinaryService:
    @staticmethod
    def upload_image(image_bytes, folder):
        try:
            result = cloudinary.uploader.upload(
                image_bytes,
                folder=folder,
                resource_type="image"
            )
            return result['secure_url']
        except Exception as e:
            current_app.logger.error(f"Error uploading to Cloudinary: {str(e)}")
            return None

    @staticmethod
    def upload_original_image(image_bytes):
        return CloudinaryService.upload_image(image_bytes, "original_uniform")

    @staticmethod
    def upload_processed_image(image_bytes):
        return CloudinaryService.upload_image(image_bytes, "processed_uniform")
