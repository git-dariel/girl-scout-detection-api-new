import cv2
import numpy as np
from flask import current_app
from app.models.model_loader import ModelLoader

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"]