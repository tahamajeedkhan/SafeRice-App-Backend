from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os
import io
from PIL import Image

# Nutritional constants per 100 grams of rice
NUTRITIONAL_INFO_PER_100G = {
    'short': {'calories': 130, 'protein': 2.7, 'fat': 0.3, 'carbs': 28.2},
    'medium': {'calories': 129, 'protein': 2.6, 'fat': 0.4, 'carbs': 27.8},
    'long': {'calories': 130, 'protein': 2.7, 'fat': 0.3, 'carbs': 28.2}
}

# Rice density in g/cm³ (average)
RICE_DENSITY = 1.45  # g/cm³

# Conversion factor from pixels to cm (you can adjust this based on your setup)
PIXEL_TO_CM = 0.01  # Assumed 1 pixel = 0.01 cm

app = Flask(__name__)

def classify_rice_type(length_cm, width_cm):
    """
    Classify rice grain based on the length-to-width ratio.
    """
    ratio = length_cm / width_cm
    if ratio <= 2:
        return 'short'
    elif 2 < ratio < 3:
        return 'medium'
    else:
        return 'long'

def calculate_rice_weight(length_cm, width_cm):
    """
    Calculate the weight of a rice grain using an ellipsoid approximation.
    """
    volume_cm3 = (4/3) * np.pi * (length_cm / 2) * (width_cm / 2)**2
    weight_g = volume_cm3 * RICE_DENSITY
    return weight_g, volume_cm3

def calculate_nutritional_value(weight_g, rice_type):
    """
    Calculate nutritional values based on the weight of the rice grain and type.
    """
    nutrition = NUTRITIONAL_INFO_PER_100G[rice_type]
    calories = (weight_g / 100) * nutrition['calories']
    protein = (weight_g / 100) * nutrition['protein']
    fat = (weight_g / 100) * nutrition['fat']
    carbs = (weight_g / 100) * nutrition['carbs']

    return {
        'weight': weight_g,
        'calories': calories,
        'protein': protein,
        'fat': fat,
        'carbs': carbs,
        'type': rice_type
    }

def extract_rice_features(image):
    if image is None:
        return [], None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rice_features = []
    total_weight = 0
    dominant_type_count = {"short": 0, "medium": 0, "long": 0}
    avg_aspect_ratio = 0
    valid_contours = 0

    # Create a copy of the original image to draw contours on
    image_with_contours = image.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filter small contours
            if len(contour) >= 5:  # Needed for ellipse fitting
                # Draw contour on the image
                cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 2)
                
                # Fit ellipse and extract parameters
                ellipse = cv2.fitEllipse(contour)
                (x, y), (MA, ma), angle = ellipse
                
                # Convert axis lengths to cm
                length_px = max(MA, ma)
                width_px = min(MA, ma)
                length_cm = length_px * PIXEL_TO_CM
                width_cm = width_px * PIXEL_TO_CM
                
                # Calculate aspect ratio
                aspect_ratio = length_cm / width_cm
                avg_aspect_ratio += aspect_ratio
                valid_contours += 1

                # Classify and calculate metrics
                rice_type = classify_rice_type(length_cm, width_cm)
                dominant_type_count[rice_type] += 1
                weight, volume = calculate_rice_weight(length_cm, width_cm)
                total_weight += weight
                
                # Store features
                rice_features.append({
                    'length_cm': length_cm,
                    'width_cm': width_cm,
                    'aspect_ratio': aspect_ratio,
                    'volume_cm3': volume,
                    'weight_g': weight,
                    'type': rice_type
                })

    # Calculate the most common grain type
    grain_type = max(dominant_type_count, key=dominant_type_count.get)
    
    # Calculate average aspect ratio
    if valid_contours > 0:
        avg_aspect_ratio = avg_aspect_ratio / valid_contours
    else:
        avg_aspect_ratio = 0

    return rice_features, image_with_contours, len(rice_features), grain_type, avg_aspect_ratio, total_weight

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    # Read image file into memory
    img_bytes = file.read()
    
    # Convert to OpenCV format
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    # Process the image
    features, image_with_contours, grain_count, grain_type, aspect_ratio, total_weight = extract_rice_features(image)
    
    # Convert the image with contours to base64 for response
    _, buffer = cv2.imencode('.jpg', image_with_contours)
    outlined_image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Round values for better display
    aspect_ratio = round(aspect_ratio, 2)
    total_weight = round(total_weight, 2)
    
    # Prepare response
    response = {
        'grain_count': grain_count,
        'grain_type': grain_type,
        'aspect_ratio': aspect_ratio,
        'total_weight': total_weight,
        'outlined_image': outlined_image_base64
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)