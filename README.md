# Girl Scout Uniform Detection API

A Flask-based REST API for detecting and verifying authentic Girl Scout uniforms using TensorFlow and computer vision techniques.

## Overview

This API provides endpoints for analyzing images of Girl Scout uniforms to determine their authenticity. The system uses a custom-trained TensorFlow model to classify uniforms as either authentic or fake, providing detailed explanations for its decisions.

## Features

- Image upload and processing
- Uniform authenticity detection with confidence scores
- Detailed explanations of authentication results
- Visual feedback with prediction overlays
- Secure cloud storage of processed images
- Environment-based configuration

## Tech Stack

- Python 3.x
- TensorFlow for model inference
- OpenCV for image processing
- Flask for REST API
- Cloudinary for image storage
- Swagger/Flasgger for API documentation

## Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)
- Cloudinary account for image storage

## Installation

1. Clone the repository:

```bash
git clone https://github.com/git-dariel/girl-scout-detection-api.git
cd girl-scout-detection-api
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   Create a `.env` file in the root directory with the following:

```env
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

## API Endpoints

### POST /api/detect-uniform

Analyzes an uploaded image of a Girl Scout uniform.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image file)

**Response:**
```json
{
    "is_authentic": true,
    "confidence_score": 95.5,
    "message": "This appears to be an authentic Girl Scout uniform with high confidence.",
    "details": [
        "Official GSP emblem placement and stitching pattern",
        "Correct shade of green matching official specifications",
        "Proper collar design and yellow neckerchief placement",
        "Quality of fabric and pattern matches official standards",
        "Authentic stitching pattern and seam construction"
    ],
    "original_image_url": "http://example.com/original.png",
    "processed_image_url": "http://example.com/processed.png"
}
```

## Model Information

The API uses a TensorFlow model trained on a dataset of authentic and counterfeit Girl Scout uniforms. The model analyzes various aspects of the uniform including:
- GSP emblem placement and quality
- Fabric color and pattern
- Stitching patterns
- Collar and neckerchief design
- Overall construction quality

## Running the Application

1. Ensure all environment variables are set
2. Run the Flask application:

```bash
python run.py
```

The API will be available at `http://localhost:5000`

## API Documentation

Access the Swagger documentation at `http://localhost:5000/apidocs/` when the application is running.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
