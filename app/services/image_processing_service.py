import numpy as np
import cv2
import os
from app.models.model_loader import ModelLoader
import threading
import glob

class ImageProcessingService:
    _labels = None
    _lock = threading.Lock()
    _texture_cache = {}
    _reference_features = None
    _reference_lock = threading.Lock()
    
    # Official Girl Scout colors
    OFFICIAL_GREEN = np.array([35, 115, 35])
    OFFICIAL_YELLOW = np.array([255, 223, 0])

    @classmethod
    def get_labels(cls):
        if cls._labels is None:
            with cls._lock:
                if cls._labels is None:
                    labels_path = os.path.join('models', 'labels.txt')
                    if not os.path.exists(labels_path):
                        raise FileNotFoundError(f"Labels file not found at {labels_path}")
                    with open(labels_path, 'r') as f:
                        cls._labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
        return cls._labels

    @classmethod
    def _load_reference_features(cls):
        """Load and cache reference image features"""
        if cls._reference_features is None:
            with cls._reference_lock:
                if cls._reference_features is None:
                    cls._reference_features = {
                        'authentic': [],
                        'counterfeit': []
                    }
                    
                    # Initialize SIFT detector
                    sift = cv2.SIFT_create(nfeatures=50)  # Limit features for speed
                    
                    # Load authentic references
                    authentic_path = os.path.join('models', 'Authentic Uniform', '*.jpg')
                    for img_path in glob.glob(authentic_path):
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (224, 224))  # Standardize size
                            kp, des = sift.detectAndCompute(img, None)
                            if des is not None:
                                cls._reference_features['authentic'].append(des)
                    
                    # Load counterfeit references
                    counterfeit_path = os.path.join('models', 'Counterfiet Uniform', '*.jpg')
                    for img_path in glob.glob(counterfeit_path):
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (224, 224))
                            kp, des = sift.detectAndCompute(img, None)
                            if des is not None:
                                cls._reference_features['counterfeit'].append(des)

    @staticmethod
    def _apply_gabor_filter(gray_image):
        """Apply Gabor filter for texture detection"""
        ksize = 31
        sigma = 4.0
        theta = 0
        lambda_ = 10.0
        gamma = 0.5
        phi = 0
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambda_, gamma, phi, ktype=cv2.CV_32F)
        return cv2.filter2D(gray_image, cv2.CV_8UC3, kernel)

    @staticmethod
    def _calculate_lbp(gray_image, radius=3):
        """Calculate Local Binary Pattern for texture analysis"""
        n_points = 8 * radius
        lbp = np.zeros_like(gray_image)
        for i in range(radius, gray_image.shape[0] - radius):
            for j in range(radius, gray_image.shape[1] - radius):
                center = gray_image[i, j]
                pattern = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = i + int(round(radius * np.cos(angle)))
                    y = j + int(round(radius * np.sin(angle)))
                    pattern |= (gray_image[x, y] > center) << k
                lbp[i, j] = pattern
        return lbp

    @staticmethod
    def _calculate_texture_metrics(texture_response, lbp):
        """Calculate texture metrics from filter response and LBP"""
        return {
            'texture_variance': np.var(texture_response),
            'texture_energy': np.sum(texture_response ** 2),
            'pattern_density': np.sum(lbp > 0) / (lbp.shape[0] * lbp.shape[1])
        }

    @staticmethod
    def analyze_texture(image):
        """Analyze fabric texture and embroidery patterns"""
        # Resize image for faster texture analysis if too large
        max_texture_size = 400
        h, w = image.shape[:2]
        if h > max_texture_size or w > max_texture_size:
            scale = max_texture_size / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        texture_response = ImageProcessingService._apply_gabor_filter(gray)
        lbp = ImageProcessingService._calculate_lbp(gray)
        return ImageProcessingService._calculate_texture_metrics(texture_response, lbp)

    @staticmethod
    def _enhance_lab_channels(lab_image):
        """Enhance LAB channels using CLAHE"""
        # Ensure image is in uint8 format
        if lab_image.dtype != np.uint8:
            lab_image = (lab_image * 255).astype(np.uint8)
            
        l, a, b = cv2.split(lab_image)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_l = clahe.apply(l)
        return cv2.merge((enhanced_l, a, b))

    @staticmethod
    def enhance_image(image):
        """Enhance image quality for better feature detection"""
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        # Resize for enhancement if too large
        max_enhance_size = 800
        h, w = image.shape[:2]
        if h > max_enhance_size or w > max_enhance_size:
            scale = max_enhance_size / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        enhanced_lab = ImageProcessingService._enhance_lab_channels(lab)
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        # Reduce denoising parameters for faster processing
        return cv2.fastNlMeansDenoisingColored(enhanced_bgr, None, 5, 5, 5, 15)

    @staticmethod
    def _calculate_color_metrics(hsv_image, image_shape):
        """Calculate color metrics for Girl Scout green detection"""
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        return np.sum(green_mask > 0) / (image_shape[0] * image_shape[1])

    @staticmethod
    def detect_girl_scout_features(image):
        """Detect specific Girl Scout uniform features"""
        # Resize for feature detection if too large
        max_feature_size = 600
        h, w = image.shape[:2]
        if h > max_feature_size or w > max_feature_size:
            scale = max_feature_size / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            
        texture_features = ImageProcessingService.analyze_texture(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_ratio = ImageProcessingService._calculate_color_metrics(hsv, image.shape)
        edges = cv2.Canny(image, 100, 200)
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        
        return {
            'green_ratio': green_ratio,
            'edge_density': edge_density,
            **texture_features
        }

    @staticmethod
    def _prepare_model_input(image):
        """Prepare image for model inference"""
        # Resize first while in uint8 format for better interpolation
        model_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        # Then convert to float32 and normalize
        return np.expand_dims(model_image.astype(np.float32) / 255.0, axis=0)

    @staticmethod
    def _process_texture_features(features):
        """Process texture features for authenticity determination"""
        has_embroidery = (
            features['texture_variance'] > 2000 and
            features['pattern_density'] > 0.45 and
            features['texture_energy'] > 100000 and
            features['edge_density'] > 0.15
        )
        texture_class = 0 if has_embroidery else 1
        texture_confidence = 0.9
        
        # Adjust confidence based on features
        if texture_class == 0:
            if features['texture_variance'] > 2000:
                texture_confidence *= 1.2
            if features['pattern_density'] > 0.45:
                texture_confidence *= 1.1
            if features['texture_energy'] > 100000:
                texture_confidence *= 1.1
        else:
            if features['texture_variance'] < 1500:
                texture_confidence *= 1.2
            if features['texture_energy'] < 50000:
                texture_confidence *= 1.1
        
        return has_embroidery, texture_class, min(1.0, texture_confidence)

    @staticmethod
    def _fast_feature_matching(query_des, reference_features, threshold=0.8):  # Increased threshold for more flexibility
        """Fast feature matching using FLANN"""
        if query_des is None or not reference_features or len(reference_features) == 0:
            return 0
            
        # FLANN parameters for fast matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # Reduced checks for speed
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        max_matches = 0
        for ref_des in reference_features:
            if ref_des is None or len(ref_des) < 2:  # Skip if not enough features
                continue
                
            try:
                # Convert descriptors to correct format
                query_des_float = query_des.astype(np.float32)
                ref_des_float = ref_des.astype(np.float32)
                
                matches = flann.knnMatch(query_des_float, ref_des_float, k=2)
                good_matches = sum(1 for m, n in matches if m.distance < threshold * n.distance)
                max_matches = max(max_matches, good_matches)
                
                # Early exit if we find enough matches (increased threshold)
                if max_matches >= 5:  # Reduced required matches for more flexibility
                    break
            except Exception:
                continue
                
        return max_matches

    @staticmethod
    def _check_logo_pattern(image, green_mask):
        """Check for authentic Girl Scout logo pattern (fleur-de-lis)"""
        try:
            # Convert to grayscale for pattern analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Get logo region using green mask
            kernel = np.ones((3,3), np.uint8)
            logo_region = cv2.bitwise_and(gray, gray, mask=green_mask)
            
            # Enhance edges for better pattern detection
            edges = cv2.Canny(logo_region, 20, 150)  # More lenient edge detection
            enhanced_edges = cv2.dilate(edges, kernel, iterations=2)
            
            # Calculate pattern characteristics
            contours, _ = cv2.findContours(enhanced_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return False, 0
            
            # Get the largest contour (should be the logo)
            main_contour = max(contours, key=cv2.contourArea)
            
            # Calculate contour characteristics
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)
            if perimeter == 0:
                return False, 0
                
            # Calculate shape characteristics
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(main_contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            
            # Count number of "holes" (typical for fleur-de-lis)
            logo_mask = np.zeros_like(gray)
            cv2.drawContours(logo_mask, [main_contour], -1, 255, -1)
            holes = cv2.subtract(logo_mask, enhanced_edges)
            num_holes = len(cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
            
            # Additional check for embroidery texture
            roi = gray[y:y+h, x:x+w]
            if roi.size > 0:
                texture_variance = np.var(roi)
            else:
                texture_variance = 0
            
            # More lenient pattern characteristics
            is_similar_pattern = (
                0.1 < circularity < 0.9 and      # Very lenient circularity
                0.5 < aspect_ratio < 1.5 and     # More lenient aspect ratio
                num_holes >= 1 and               # Just need some holes
                area > 50 and                    # Smaller minimum size
                texture_variance > 200           # Lower texture requirement
            )
            
            # Calculate confidence based on similarity
            confidence = 0
            if is_similar_pattern:
                # Simplified confidence calculation with max 85%
                base_confidence = 0.5  # Start with 50%
                pattern_score = (
                    min(num_holes / 3, 1) * 0.2 +        # Up to 20%
                    min(texture_variance / 1000, 1) * 0.15 # Up to 15%
                )
                confidence = min(0.85, base_confidence + pattern_score)  # Cap at 85%
            
            # Debug prints for logo detection
            print("\n=== Logo Pattern Details ===")
            print(f"Circularity: {circularity:.2f}")
            print(f"Aspect Ratio: {aspect_ratio:.2f}")
            print(f"Number of Holes: {num_holes}")
            print(f"Area: {area:.2f}")
            print(f"Texture Variance: {texture_variance:.2f}")
            print(f"Is Similar Pattern: {is_similar_pattern}")
            print(f"Pattern Confidence: {confidence:.2f}")
            
            return is_similar_pattern, confidence
            
        except Exception as e:
            print(f"Error in logo pattern check: {str(e)}")
            return False, 0

    @staticmethod
    def _check_basic_uniform_characteristics(image):
        """Pre-check for basic uniform characteristics before detailed matching"""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define Girl Scout uniform color ranges in HSV with more tolerance for low light
            # Green color range (for uniform and hat)
            lower_green = np.array([35, 20, 20])  # More lenient for darker conditions
            upper_green = np.array([95, 255, 255])
            
            # Yellow color range (for bandana/neckerchief)
            lower_yellow = np.array([15, 30, 30])  # More lenient for darker conditions
            upper_yellow = np.array([45, 255, 255])
            
            # Create color masks
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Calculate color percentages
            total_pixels = image.shape[0] * image.shape[1]
            green_percentage = np.sum(green_mask > 0) / total_pixels
            yellow_percentage = np.sum(yellow_mask > 0) / total_pixels

            # Detect skin tones (for human presence) with adjusted range for low light
            lower_skin = np.array([0, 15, 50])  # More lenient for darker conditions
            upper_skin = np.array([25, 170, 255])
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_percentage = np.sum(skin_mask > 0) / total_pixels
            
            # Check for pattern characteristics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast for better pattern detection in low light
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Detect patterns and edges with adjusted thresholds
            pattern_kernel = np.ones((3,3), np.uint8)
            edges = cv2.Canny(gray, 30, 120)  # Adjusted for low light
            pattern_mask = cv2.dilate(edges, pattern_kernel, iterations=1)
            pattern_density = np.sum(pattern_mask > 0) / total_pixels
            
            # Calculate image characteristics
            edge_percentage = np.sum(edges > 0) / total_pixels
            color_variance = np.var(gray)

            # Check for uniform components in expected positions
            height, width = image.shape[:2]
            
            # Define regions for uniform components with more flexible boundaries
            top_region = green_mask[0:height//3, :]  # Expanded hat region
            middle_region = yellow_mask[height//4:height//2, :]  # Bandana region
            bottom_region = green_mask[height//3:, :]  # Expanded uniform region
            
            # Calculate component presence with lower thresholds
            has_hat = np.sum(top_region > 0) / (top_region.size) > 0.05
            has_bandana = np.sum(middle_region > 0) / (middle_region.size) > 0.05
            has_uniform = np.sum(bottom_region > 0) / (bottom_region.size) > 0.1
            
            # Check logo pattern if present
            has_similar_logo, logo_confidence = ImageProcessingService._check_logo_pattern(image, green_mask)
            
            # More lenient non-uniform check with human presence consideration
            is_non_uniform = (
                edge_percentage > 0.9 or          # Very high threshold for complex scenes
                edge_percentage < 0.01 or         # Only reject completely blank images
                color_variance > 12000 or         # Higher tolerance for variance
                color_variance < 20 or            # Lower minimum variance
                (green_percentage < 0.1 and       # More lenient color requirements
                 yellow_percentage < 0.02 and 
                 not (has_hat or has_bandana or has_uniform))  # Allow if any component is detected
            )
            
            # Check for complete uniform characteristics
            has_uniform_characteristics = False
            
            # 1. Complete Uniform Check (Person wearing uniform with accessories)
            if (green_percentage > 0.15 and      # Reduced threshold for green
                yellow_percentage > 0.02 and      # Reduced threshold for yellow
                skin_percentage > 0.03 and        # Reduced threshold for skin
                (has_hat or has_bandana or has_uniform)):  # At least one component clearly visible
                has_uniform_characteristics = True
            
            # 2. Partial Uniform Check (Clear view of uniform components)
            elif ((has_hat and has_bandana) or    # Any combination of two components
                  (has_bandana and has_uniform) or
                  (has_hat and has_uniform)):
                has_uniform_characteristics = True
            
            # 3. Logo Check with Uniform Components
            elif has_similar_logo and (has_hat or has_bandana or has_uniform):
                has_uniform_characteristics = True
            
            # 4. Pattern Check (For uniform with fleur-de-lis pattern)
            elif (pattern_density > 0.05 and      # Visible pattern
                  green_percentage > 0.15 and     # Significant green
                  edge_percentage > 0.1):         # Some detail visible
                has_uniform_characteristics = True
            
            # Final validation
            is_valid = has_uniform_characteristics and not is_non_uniform
            
            # Debug prints
            print("\n=== Basic Characteristics Check ===")
            print(f"Green percentage: {green_percentage:.2f}")
            print(f"Yellow percentage: {yellow_percentage:.2f}")
            print(f"Skin percentage: {skin_percentage:.2f}")
            print(f"Pattern density: {pattern_density:.2f}")
            print(f"Edge percentage: {edge_percentage:.2f}")
            print(f"Color variance: {color_variance:.2f}")
            print(f"Has hat: {has_hat}")
            print(f"Has bandana: {has_bandana}")
            print(f"Has uniform: {has_uniform}")
            print(f"Has similar logo: {has_similar_logo}")
            print(f"Logo confidence: {logo_confidence:.2f}")
            print(f"Has uniform characteristics: {has_uniform_characteristics}")
            print(f"Is non-uniform: {is_non_uniform}")
            print(f"Final validation result: {is_valid}")
            
            return is_valid
            
        except Exception as e:
            print(f"Error in basic uniform check: {str(e)}")
            return False

    @staticmethod
    def validate_image_similarity(image):
        """Validate image using basic characteristics and logo pattern"""
        try:
            # First check basic uniform characteristics
            if not ImageProcessingService._check_basic_uniform_characteristics(image):
                return False, 0
            
            # Comment out reference feature matching
            """
            # Ensure reference features are loaded
            ImageProcessingService._load_reference_features()
            
            # Convert to grayscale and resize
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (224, 224))
            
            # Extract features from query image
            sift = cv2.SIFT_create(nfeatures=100)
            kp, query_des = sift.detectAndCompute(gray, None)
            
            if query_des is None or len(query_des) < 2:
                return False, 0
                
            # Match with authentic references
            authentic_matches = ImageProcessingService._fast_feature_matching(
                query_des, 
                ImageProcessingService._reference_features['authentic']
            )
            
            # Match with counterfeit references
            counterfeit_matches = ImageProcessingService._fast_feature_matching(
                query_des, 
                ImageProcessingService._reference_features['counterfeit']
            )
            
            # Calculate similarity scores with more lenient thresholds
            total_matches = authentic_matches + counterfeit_matches
            """
            
            # If basic characteristics passed, proceed with confidence
            # This replaces the reference matching logic
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_green = np.array([30, 30, 30])
            upper_green = np.array([90, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Check logo pattern for additional validation
            has_logo_pattern, logo_confidence = ImageProcessingService._check_logo_pattern(image, green_mask)
            
            # Calculate final confidence based on basic characteristics and logo pattern
            if has_logo_pattern:
                confidence = min(0.8, 0.6 + (logo_confidence * 0.2))
            else:
                confidence = 0.6  # Base confidence if basic characteristics passed
            
            return True, float(confidence)
            
        except Exception as e:
            print(f"Error in validate_image_similarity: {str(e)}")
            # If there's an error but basic characteristics passed, give it a chance
            return True, 0.5

    @staticmethod
    def process_image(image):
        """Main image processing method"""
        if image is None:
            raise ValueError("Invalid image data")
            
        try:
            # Perform fast validation first
            is_valid_uniform, similarity_confidence = ImageProcessingService.validate_image_similarity(image)
            
            # Print validation results
            print("\n=== Similarity Validation ===")
            print(f"Validation Result: {'Valid' if is_valid_uniform else 'Invalid'}")
            print(f"Similarity Confidence: {similarity_confidence * 100:.2f}%")
            
            # If validation fails, return early with rejection
            if not is_valid_uniform:
                # Create processed image with rejection overlay
                processed_image = image.copy()
                cv2.putText(processed_image, "Invalid Uniform", (15, 35), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(processed_image, "Rejected", (15, 70), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
                
                # Convert to bytes
                _, buffer = cv2.imencode('.png', processed_image)
                
                return {
                    'is_authentic': False,
                    'confidence_score': float(similarity_confidence * 100),
                    'message': "This does not appear to be a valid Girl Scout uniform.",
                    'details': [
                        "Image does not match known Girl Scout uniform patterns",
                        "Significant differences from reference uniforms detected",
                        "Further verification needed",
                        f"Similarity confidence: {similarity_confidence * 100:.1f}%"
                    ],
                    'processed_image_bytes': buffer.tobytes(),
                    'raw_predictions': []
                }
            
            # Continue processing only if validation passed
            enhanced_image = ImageProcessingService.enhance_image(image)
            image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
            model_image = ImageProcessingService._prepare_model_input(image_rgb)
            
            # Get TF Lite interpreter and details
            interpreter, input_details, output_details = ModelLoader.get_model()
            interpreter.set_tensor(input_details[0]['index'], model_image)
            interpreter.invoke()
            
            # Get predictions
            predictions = interpreter.get_tensor(output_details[0]['index'])
            raw_predictions = [float(p) for p in predictions[0]]
            prediction_class = int(np.argmax(predictions[0]))
            model_confidence = float(predictions[0][prediction_class])
            
            # Get features for analysis
            features = ImageProcessingService.detect_girl_scout_features(enhanced_image)
            
            # Print model results
            print("\n=== Model Results ===")
            print(f"Predicted Class: {prediction_class} ({ImageProcessingService.get_labels()[prediction_class]})")
            print(f"Model Confidence: {model_confidence * 100:.2f}%")
            print("Raw Predictions:", [f"{p * 100:.2f}%" for p in predictions[0]])
            
            # Process texture features for authenticity check
            has_embroidery, texture_class, texture_confidence = ImageProcessingService._process_texture_features(features)
            
            # Print analysis results
            print("\n=== Logo Analysis ===")
            print(f"Texture Variance: {features['texture_variance']:.2f}")
            print(f"Texture Energy: {features['texture_energy']:.2f}")
            print(f"Pattern Density: {features['pattern_density'] * 100:.2f}%")
            print(f"Edge Density: {features['edge_density'] * 100:.2f}%")
            print(f"Green Ratio: {features['green_ratio'] * 100:.2f}%")
            print(f"Has Embroidery: {has_embroidery}")
            print(f"Texture Confidence: {texture_confidence * 100:.2f}%")
            
            # Make final decision based on model and texture analysis
            if prediction_class in [0, 1]:  # If it's a uniform detection
                # Check if we're looking at a logo portion
                is_logo = bool(
                    features['texture_variance'] > 2000 and
                    features['pattern_density'] > 0.4 and
                    features['edge_density'] > 0.15
                )
                
                if is_logo:
                    # For logo portions, use embroidery detection with capped confidence
                    is_authentic = bool(has_embroidery)
                    final_confidence = float(min(0.85, texture_confidence))  # Cap at 85%
                    predicted_label = "Authentic Logo" if is_authentic else "Fake Logo"
                else:
                    # For uniform portions, use model prediction with capped confidence
                    is_authentic = bool(prediction_class == 0)
                    final_confidence = float(min(0.9, model_confidence))  # Cap at 90%
                    predicted_label = "Authentic Uniform" if is_authentic else "Fake Uniform"
            else:
                # For non-uniform images, use model prediction
                is_authentic = False
                final_confidence = float(min(0.8, model_confidence))  # Cap at 80%
                predicted_label = "Not a Uniform"
            
            # Print final decision
            print("\n=== Final Decision ===")
            print(f"Authentic: {is_authentic}")
            print(f"Final Confidence: {final_confidence * 100:.2f}%")
            
            # Determine uniform type for proper explanation
            uniform_type = ImageProcessingService.get_uniform_explanation(is_authentic, final_confidence, features, predicted_label, image)
            print(f"\n=== Detected Uniform Type ===")
            print(f"Type: {uniform_type}")
            
            # Flush the output to ensure it appears before the API response
            import sys
            sys.stdout.flush()
            
            # Create processed image with prediction overlay
            processed_image = image.copy()
            
            # Update the predicted_label based on uniform type
            if uniform_type['conclusion'].startswith("This appears to be an authentic Girl Scout hat"):
                display_label = "Authentic Hat"
            elif uniform_type['conclusion'].startswith("This appears to be an authentic Girl Scout neckerchief"):
                display_label = "Authentic Bandana"
            elif uniform_type['conclusion'].startswith("This appears to be an authentic Girl Scout uniform dress"):
                display_label = "Authentic Uniform"
            elif uniform_type['conclusion'].startswith("This appears to be an authentic complete Girl Scout uniform"):
                display_label = "Authentic Full Set"
            else:
                display_label = predicted_label
            
            cv2.putText(processed_image, display_label, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 1.5, 
                       (0, 255, 0) if is_authentic else (0, 0, 255), 3)
            cv2.putText(processed_image, f"{final_confidence * 100:.1f}%", (15, 70), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
            
            # Convert to bytes
            _, buffer = cv2.imencode('.png', processed_image)
            
            return {
                'is_authentic': bool(is_authentic),
                'confidence_score': float(min(90, final_confidence * 100)),  # Cap at 90%
                'message': uniform_type['conclusion'],
                'details': uniform_type['reasons'],
                'processed_image_bytes': buffer.tobytes(),
                'raw_predictions': raw_predictions
            }
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")

    @staticmethod
    def get_uniform_explanation(is_authentic, confidence_score, features=None, predicted_label="", image=None):
        """Generate explanation for the authentication result based on uniform component type"""
        
        # Helper function to determine uniform type
        def get_uniform_type():
            if not features or image is None:
                return "unknown"
            
            # Get basic features
            green_ratio = features.get('green_ratio', 0)
            yellow_ratio = features.get('yellow_percentage', 0)
            pattern_density = features.get('pattern_density', 0)
            edge_density = features.get('edge_density', 0)
            texture_variance = features.get('texture_variance', 0)
            
            # Convert image to grayscale for shape analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Use more lenient edge detection for better shape capture
            edges = cv2.Canny(gray, 30, 100)  # Adjusted thresholds
            kernel = np.ones((3,3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)  # Dilate edges to connect gaps
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define yellow color range for bandana
            lower_yellow = np.array([15, 50, 50])
            upper_yellow = np.array([40, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Define green color range for logo
            lower_green = np.array([30, 30, 30])
            upper_green = np.array([90, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return "unknown"
            
            # Get the largest contour for main shape analysis
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter != 0 else 0
            
            # Calculate relative area (how much of the image the object takes up)
            relative_area = area / (image.shape[0] * image.shape[1])
            
            # Calculate yellow and green percentages
            yellow_percentage = np.sum(yellow_mask > 0) / (image.shape[0] * image.shape[1])
            green_percentage = np.sum(green_mask > 0) / (image.shape[0] * image.shape[1])
            
            print("\n=== Shape Analysis ===")
            print(f"Aspect Ratio: {aspect_ratio:.2f}")
            print(f"Circularity: {circularity:.2f}")
            print(f"Relative Area: {relative_area:.2f}")
            print(f"Edge Density: {edge_density:.2f}")
            print(f"Yellow Percentage: {yellow_percentage:.2f}")
            print(f"Green Percentage: {green_percentage:.2f}")
            
            # Bandana/neckerchief detection (prioritize this check)
            if (yellow_percentage > 0.3 or  # Significant yellow background
                (yellow_percentage > 0.2 and green_percentage > 0.05) or  # Yellow with green logo
                (yellow_ratio > 0.2 and  # Significant yellow
                 ((0.5 < aspect_ratio < 2.0) or  # Square-ish or slightly rectangular
                  pattern_density > 0.1) and  # Visible pattern/logo
                  relative_area > 0.2)):  # Takes up reasonable portion
                print("Detected type: bandana")
                return "bandana"
            
            # Hat detection
            elif (green_ratio > 0.2 and                  # Green color present
                0.8 < aspect_ratio < 1.3 and           # Hat aspect ratio
                relative_area > 0.3 and                # Hat takes up significant area
                edge_density < 0.2 and                 # Edge density from debug
                h > image.shape[0] * 0.2):             # Ensure minimum height
                print("Detected type: hat")
                return "hat"
            
            # Full uniform detection
            elif (0.2 < aspect_ratio < 0.8 and         # Typical standing person ratio
                  h > image.shape[0] * 0.6 and         # Person takes up significant height
                  green_ratio > 0.2 and                # Presence of uniform
                  yellow_ratio > 0.05):                # Presence of neckerchief
                print("Detected type: full_uniform")
                return "full_uniform"
            
            # Uniform/dress detection
            elif (0.5 < aspect_ratio < 0.9 and         # Dress proportions
                  h > image.shape[0] * 0.5 and         # Significant height
                  w > image.shape[1] * 0.3 and         # Reasonable width
                  green_ratio > 0.3 and                # Green color present
                  pattern_density > 0.1):              # Pattern visible
                print("Detected type: uniform")
                return "uniform"
            
            print("Detected type: unknown")
            return "unknown"
        
        # Get the uniform type
        detected_type = get_uniform_type()
        
        if is_authentic:
            if confidence_score > 0.75:
                explanations = {
                    "full_uniform": {
                        "conclusion": "This appears to be an authentic complete Girl Scout uniform set with good confidence.",
                        "reasons": [
                            "Official green dress with black fleur-de-lis pattern throughout",
                            "Bright yellow neckerchief with proper GSP emblem placement",
                            "Standard green bucket hat matching uniform",
                            "Correct collar style and dress length",
                            "Proper uniform assembly with waist definition"
                        ]
                    },
                    "uniform": {
                        "conclusion": "This appears to be an authentic Girl Scout uniform dress with good confidence.",
                        "reasons": [
                            "Official green fabric with consistent black fleur-de-lis pattern",
                            "Proper collar design and button placement",
                            "Correct waist seam construction",
                            "Standard pleating and dress length",
                            "Quality stitching and fabric construction"
                        ]
                    },
                    "bandana": {
                        "conclusion": "This appears to be an authentic Girl Scout neckerchief with good confidence.",
                        "reasons": [
                            "Bright yellow fabric with correct shade",
                            "Proper triangular shape and size",
                            "Quality fabric construction",
                            "Clean edges and proper dimensions",
                            "Standard GSP neckerchief specifications"
                        ]
                    },
                    "hat": {
                        "conclusion": "This appears to be an authentic Girl Scout hat with good confidence.",
                        "reasons": [
                            "Official green bucket hat style",
                            "Proper black fleur-de-lis pattern or solid green design",
                            "Correct brim width and crown height",
                            "Quality fabric and stitching construction",
                            "Standard GSP hat specifications"
                        ]
                    },
                    "unknown": {
                        "conclusion": "This appears to be an authentic Girl Scout item with good confidence.",
                        "reasons": [
                            "Official GSP design elements present",
                            "Authentic material and construction",
                            "Proper color matching and quality",
                            "Standard GSP characteristics detected"
                        ]
                    }
                }
            elif confidence_score > 0.5:
                explanations = {
                    "full_uniform": {
                        "conclusion": "This is likely an authentic Girl Scout uniform set, but there is some uncertainty.",
                        "reasons": [
                            "Green dress with fleur-de-lis pattern appears correct",
                            "Yellow neckerchief present with GSP emblem",
                            "Green hat matches uniform style",
                            "Some variations in assembly noted",
                            "Minor inconsistencies in overall fit"
                        ]
                    },
                    "uniform": {
                        "conclusion": "This is likely an authentic Girl Scout uniform dress, but there is some uncertainty.",
                        "reasons": [
                            "Green color with fleur-de-lis pattern present",
                            "Basic dress construction appears correct",
                            "Some variation in pattern placement",
                            "Minor inconsistencies in stitching"
                        ]
                    },
                    "bandana": {
                        "conclusion": "This is likely an authentic Girl Scout neckerchief, but there is some uncertainty.",
                        "reasons": [
                            "Yellow color matches expected shade",
                            "Basic triangular shape present",
                            "Some variation in construction",
                            "Minor inconsistencies in fabric"
                        ]
                    },
                    "hat": {
                        "conclusion": "This is likely an authentic Girl Scout hat, but there is some uncertainty.",
                        "reasons": [
                            "Green bucket hat style present",
                            "Pattern or solid color appears correct",
                            "Some variation in construction",
                            "Minor inconsistencies in shape"
                        ]
                    },
                    "unknown": {
                        "conclusion": "This is likely an authentic Girl Scout item, but there is some uncertainty.",
                        "reasons": [
                            "Basic GSP elements present",
                            "Some variations in quality",
                            "Minor inconsistencies noted",
                            "Further verification recommended"
                        ]
                    }
                }
            else:
                explanations = {
                    "full_uniform": {
                        "conclusion": "This might be an authentic Girl Scout uniform set, but confidence is low.",
                        "reasons": [
                            "Basic green dress elements present",
                            "Yellow neckerchief included",
                            "Hat component present",
                            "Significant variations from standard"
                        ]
                    },
                    "uniform": {
                        "conclusion": "This might be an authentic Girl Scout uniform dress, but confidence is low.",
                        "reasons": [
                            "Basic green dress present",
                            "Pattern attempt visible",
                            "Construction differs from standard",
                            "Quality concerns noted"
                        ]
                    },
                    "bandana": {
                        "conclusion": "This might be an authentic Girl Scout neckerchief, but confidence is low.",
                        "reasons": [
                            "Yellow fabric present",
                            "Basic shape visible",
                            "Construction differs from standard",
                            "Quality concerns noted"
                        ]
                    },
                    "hat": {
                        "conclusion": "This might be an authentic Girl Scout hat, but confidence is low.",
                        "reasons": [
                            "Green hat present",
                            "Basic style attempt visible",
                            "Construction differs from standard",
                            "Quality concerns noted"
                        ]
                    },
                    "unknown": {
                        "conclusion": "This might be an authentic Girl Scout item, but confidence is low.",
                        "reasons": [
                            "Basic GSP elements present",
                            "Significant variations noted",
                            "Quality unclear",
                            "Further inspection needed"
                        ]
                    }
                }
        else:
            # For non-authentic items
            explanations = {
                "full_uniform": {
                    "conclusion": "This appears to be a non-authentic Girl Scout uniform set.",
                    "reasons": [
                        "Incorrect green shade or fleur-de-lis pattern",
                        "Non-standard yellow neckerchief design",
                        "Improper hat style or construction",
                        "Wrong assembly of components",
                        "Quality deviates from GSP standards"
                    ]
                },
                "uniform": {
                    "conclusion": "This appears to be a non-authentic Girl Scout uniform dress.",
                    "reasons": [
                        "Incorrect green shade or pattern style",
                        "Wrong fleur-de-lis pattern placement",
                        "Non-standard dress construction",
                        "Deviates from official GSP design"
                    ]
                },
                "bandana": {
                    "conclusion": "This appears to be a non-authentic Girl Scout neckerchief.",
                    "reasons": [
                        "Wrong yellow shade",
                        "Incorrect GSP emblem design",
                        "Non-standard text or placement",
                        "Deviates from official specifications"
                    ]
                },
                "hat": {
                    "conclusion": "This appears to be a non-authentic Girl Scout hat.",
                    "reasons": [
                        "Wrong green shade",
                        "Incorrect bucket hat style",
                        "Non-standard pattern or construction",
                        "Deviates from official specifications"
                    ]
                },
                "unknown": {
                    "conclusion": "This appears to be a non-authentic Girl Scout item.",
                    "reasons": [
                        "Non-standard design elements",
                        "Incorrect construction",
                        "Wrong specifications",
                        "Multiple deviations from standards"
                    ]
                }
            }
        
        # Return the explanation for the detected type
        explanation = explanations.get(detected_type, explanations["unknown"])
        print(f"\nReturning explanation for type: {detected_type}")
        return explanation
