from flask import Blueprint, jsonify, request
from app.services.detected_uniform_service import DetectedUniformService
from datetime import datetime
from bson.errors import InvalidId
from bson import ObjectId

detected_uniform_route = Blueprint('detected_uniform_route', __name__)

@detected_uniform_route.route('/api/get/all/detected-uniforms', methods=['GET'])
def get_detected_uniforms():
    """Get all detected uniforms with pagination and filtering"""
    try:
        # Get pagination parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))

        # Get filters
        filters = {}
        if 'is_authentic' in request.args:
            filters['is_authentic'] = request.args.get('is_authentic').lower() == 'true'
        if 'uniform_type' in request.args:
            filters['uniform_type'] = request.args.get('uniform_type')
        if 'min_confidence' in request.args:
            filters['min_confidence'] = float(request.args.get('min_confidence'))
        if 'start_date' in request.args and 'end_date' in request.args:
            filters['date_range'] = {
                'start': datetime.fromisoformat(request.args.get('start_date')),
                'end': datetime.fromisoformat(request.args.get('end_date'))
            }

        result = DetectedUniformService.get_all_detected_uniforms(page, per_page, filters)
        
        # Convert uniforms to dict for JSON serialization
        result['uniforms'] = [uniform.to_dict() for uniform in result['uniforms']]
        
        return jsonify(result), 200

    except ValueError as e:
        return jsonify({'error': f'Invalid parameter: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to get detected uniforms: {str(e)}'}), 500

@detected_uniform_route.route('/api/get/detected-uniforms/<uniform_id>', methods=['GET'])
def get_detected_uniform(uniform_id):
    """Get a specific detected uniform by ID"""
    try:
        uniform = DetectedUniformService.get_detected_uniform(uniform_id)
        if not uniform:
            return jsonify({'error': 'Detected uniform not found'}), 404
        return jsonify(uniform.to_dict()), 200

    except InvalidId:
        return jsonify({'error': 'Invalid uniform ID format'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to get detected uniform: {str(e)}'}), 500

@detected_uniform_route.route('/api/update/detected-uniforms/<uniform_id>', methods=['PUT'])
def update_detected_uniform(uniform_id):
    """Update a detected uniform"""
    try:
        updates = request.get_json()
        if not updates:
            return jsonify({'error': 'No update data provided'}), 400

        uniform = DetectedUniformService.update_detected_uniform(uniform_id, updates)
        if not uniform:
            return jsonify({'error': 'Detected uniform not found'}), 404

        return jsonify(uniform.to_dict()), 200

    except InvalidId:
        return jsonify({'error': 'Invalid uniform ID format'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to update detected uniform: {str(e)}'}), 500

@detected_uniform_route.route('/api/delete/detected-uniforms/<uniform_id>', methods=['DELETE'])
def delete_detected_uniform(uniform_id):
    """Delete a detected uniform"""
    try:
        success = DetectedUniformService.delete_detected_uniform(uniform_id)
        if not success:
            return jsonify({'error': 'Detected uniform not found'}), 404
        return jsonify({'message': 'Detected uniform deleted successfully'}), 200

    except InvalidId:
        return jsonify({'error': 'Invalid uniform ID format'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to delete detected uniform: {str(e)}'}), 500

@detected_uniform_route.route('/api/get/detected-uniforms/statistics', methods=['GET'])
def get_detection_statistics():
    """Get detection statistics"""
    try:
        stats = DetectedUniformService.get_statistics()
        return jsonify(stats), 200

    except Exception as e:
        return jsonify({'error': f'Failed to get statistics: {str(e)}'}), 500 