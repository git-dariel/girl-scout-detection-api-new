import cloudinary
from os import environ

def configure_cloudinary():
    cloudinary.config(
        cloud_name=environ.get('CLOUDINARY_CLOUD_NAME'),
        api_key=environ.get('CLOUDINARY_API_KEY'),
        api_secret=environ.get('CLOUDINARY_API_SECRET'),
    )