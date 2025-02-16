from flask import request
import cv2
import numpy as np

def handle_file_upload():
    if "file" not in request.files:
        return None, "No file uploaded"
    
    file = request.files["file"]
    if not file:
        return None, "No file selected"
    
    # Read image directly from the uploaded file
    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return None, "Invalid image file"
    
    return image, None