from bson import ObjectId
from app.models.detected_uniform import DetectedUniform, detected_uniforms

class DetectedUniformRepository:
    @staticmethod
    def insert(detected_uniform):
        """Insert a new detected uniform record"""
        result = detected_uniforms.insert_one(detected_uniform.to_dict())
        return result.inserted_id

    @staticmethod
    def find_by_id(uniform_id):
        """Find a detected uniform by ID"""
        result = detected_uniforms.find_one({'_id': ObjectId(uniform_id)})
        return DetectedUniform.from_dict(result) if result else None

    @staticmethod
    def find_all(query, skip, limit):
        """Find all detected uniforms with pagination"""
        cursor = detected_uniforms.find(query) \
                                   .sort('created_at', -1) \
                                   .skip(skip) \
                                   .limit(limit)
        return [DetectedUniform.from_dict(doc) for doc in cursor]

    @staticmethod
    def count(query):
        """Count detected uniforms matching query"""
        return detected_uniforms.count_documents(query)

    @staticmethod
    def update(uniform_id, updates):
        """Update a detected uniform"""
        result = detected_uniforms.update_one(
            {'_id': ObjectId(uniform_id)},
            {'$set': updates}
        )
        return result.modified_count > 0

    @staticmethod
    def delete(uniform_id):
        """Delete a detected uniform"""
        result = detected_uniforms.delete_one({'_id': ObjectId(uniform_id)})
        return result.deleted_count > 0

    @staticmethod
    def aggregate(pipeline):
        """Aggregate detected uniforms"""
        return list(detected_uniforms.aggregate(pipeline)) 