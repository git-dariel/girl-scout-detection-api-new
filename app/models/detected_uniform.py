from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection
client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('MONGODB_DB_NAME')]
detected_uniforms = db.detected_uniforms

class DetectedUniform:
    def __init__(self, 
                 is_authentic,
                 confidence_score,
                 message,
                 details,
                 original_image_url,
                 processed_image_url,
                 graph_url=None,
                 graph_analysis=None,
                 uniform_type=None,
                 raw_predictions=None,
                 created_at=None,
                 _id=None):
        self._id = _id or ObjectId()
        self.is_authentic = is_authentic
        self.confidence_score = confidence_score
        self.message = message
        self.details = details
        self.original_image_url = original_image_url
        self.processed_image_url = processed_image_url
        self.graph_url = graph_url
        self.graph_analysis = graph_analysis
        self.uniform_type = uniform_type
        self.raw_predictions = raw_predictions
        self.created_at = created_at or datetime.utcnow()

    @staticmethod
    def from_dict(data):
        """Create DetectedUniform instance from dictionary"""
        return DetectedUniform(
            _id=data.get('_id'),
            is_authentic=data.get('is_authentic'),
            confidence_score=data.get('confidence_score'),
            message=data.get('message'),
            details=data.get('details'),
            original_image_url=data.get('original_image_url'),
            processed_image_url=data.get('processed_image_url'),
            graph_url=data.get('graph_url'),
            graph_analysis=data.get('graph_analysis'),
            uniform_type=data.get('uniform_type'),
            raw_predictions=data.get('raw_predictions'),
            created_at=data.get('created_at')
        )

    def to_dict(self):
        """Convert instance to dictionary"""
        return {
            '_id': str(self._id),
            'is_authentic': self.is_authentic,
            'confidence_score': self.confidence_score,
            'message': self.message,
            'details': self.details,
            'original_image_url': self.original_image_url,
            'processed_image_url': self.processed_image_url,
            'graph_url': self.graph_url,
            'graph_analysis': self.graph_analysis,
            'uniform_type': self.uniform_type,
            'raw_predictions': self.raw_predictions,
            'created_at': self.created_at
        } 