import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "uploads")
    ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS", "png, jpg, jpeg, bmp").split(",")