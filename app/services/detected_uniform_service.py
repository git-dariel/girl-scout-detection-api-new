from bson import ObjectId
from app.models.detected_uniform import DetectedUniform, detected_uniforms
from datetime import datetime
from app.repositories.detected_uniform_repository import DetectedUniformRepository

class DetectedUniformService:
    @staticmethod
    def create_detected_uniform(detection_result):
        """Create a new detected uniform record"""
        try:
            # Create DetectedUniform instance
            uniform = DetectedUniform(
                is_authentic=detection_result.get('is_authentic'),
                confidence_score=detection_result.get('confidence_score'),
                message=detection_result.get('message'),
                details=detection_result.get('details'),
                original_image_url=detection_result.get('original_image_url'),
                processed_image_url=detection_result.get('processed_image_url'),
                graph_url=detection_result.get('graph_url'),
                graph_analysis=detection_result.get('graph_analysis'),
                uniform_type=detection_result.get('uniform_type'),
                raw_predictions=detection_result.get('raw_predictions')
            )

            # Use repository to insert into MongoDB
            uniform._id = DetectedUniformRepository.insert(uniform)
            return uniform

        except Exception as e:
            print(f"Error creating detected uniform: {str(e)}")
            raise

    @staticmethod
    def get_detected_uniform(uniform_id):
        """Get a detected uniform by ID"""
        try:
            return DetectedUniformRepository.find_by_id(uniform_id)

        except Exception as e:
            print(f"Error getting detected uniform: {str(e)}")
            raise

    @staticmethod
    def get_all_detected_uniforms(page=1, per_page=10, filters=None):
        """Get all detected uniforms with pagination and optional filters"""
        try:
            # Initialize query
            query = {}

            # Apply filters if provided
            if filters:
                if 'is_authentic' in filters:
                    query['is_authentic'] = filters['is_authentic']
                if 'uniform_type' in filters:
                    query['uniform_type'] = filters['uniform_type']
                if 'min_confidence' in filters:
                    query['confidence_score'] = {'$gte': filters['min_confidence']}
                if 'date_range' in filters:
                    query['created_at'] = {
                        '$gte': filters['date_range']['start'],
                        '$lte': filters['date_range']['end']
                    }

            # Calculate skip value for pagination
            skip = (page - 1) * per_page

            # Use repository to get total count for pagination
            total = DetectedUniformRepository.count(query)

            # Use repository to get paginated results
            uniforms = DetectedUniformRepository.find_all(query, skip, per_page)

            return {
                'uniforms': uniforms,
                'total': total,
                'page': page,
                'per_page': per_page,
                'total_pages': (total + per_page - 1) // per_page
            }

        except Exception as e:
            print(f"Error getting detected uniforms: {str(e)}")
            raise

    @staticmethod
    def update_detected_uniform(uniform_id, updates):
        """Update a detected uniform"""
        try:
            # Ensure _id is not in updates
            if '_id' in updates:
                del updates['_id']

            # Add updated_at timestamp
            updates['updated_at'] = datetime.utcnow()

            # Use repository to update
            success = DetectedUniformRepository.update(uniform_id, updates)

            if success:
                return DetectedUniformService.get_detected_uniform(uniform_id)
            return None

        except Exception as e:
            print(f"Error updating detected uniform: {str(e)}")
            raise

    @staticmethod
    def delete_detected_uniform(uniform_id):
        """Delete a detected uniform"""
        try:
            # Use repository to delete
            return DetectedUniformRepository.delete(uniform_id)

        except Exception as e:
            print(f"Error deleting detected uniform: {str(e)}")
            raise

    @staticmethod
    def get_statistics():
        """Get detection statistics"""
        try:
            pipeline = [
                {
                    '$group': {
                        '_id': None,
                        'total_detections': {'$sum': 1},
                        'authentic_count': {
                            '$sum': {'$cond': ['$is_authentic', 1, 0]}
                        },
                        'avg_confidence': {'$avg': '$confidence_score'},
                        'uniform_types': {
                            '$addToSet': '$uniform_type'
                        }
                    }
                }
            ]

            # Use repository to aggregate
            stats = DetectedUniformRepository.aggregate(pipeline)
            if not stats:
                return {
                    'total_detections': 0,
                    'authentic_count': 0,
                    'fake_count': 0,
                    'avg_confidence': 0,
                    'uniform_types': []
                }

            stats = stats[0]
            return {
                'total_detections': stats['total_detections'],
                'authentic_count': stats['authentic_count'],
                'fake_count': stats['total_detections'] - stats['authentic_count'],
                'avg_confidence': round(stats['avg_confidence'], 2),
                'uniform_types': stats['uniform_types']
            }

        except Exception as e:
            print(f"Error getting statistics: {str(e)}")
            raise 