import cv2
import numpy as np
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt

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

def extract_rice_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return [], None, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rice_features = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (x, y), (MA, ma), angle = ellipse
                length_px = max(MA, ma)
                width_px = min(MA, ma)

                length_cm = length_px * PIXEL_TO_CM
                width_cm = width_px * PIXEL_TO_CM

                rice_type = classify_rice_type(length_cm, width_cm)
                weight, volume = calculate_rice_weight(length_cm, width_cm)
                nutrition = calculate_nutritional_value(weight, rice_type)

                rice_features.append({
                    'length_cm': length_cm,
                    'width_cm': width_cm,
                    'volume_cm3': volume,
                    'weight_g': weight,
                    'nutrition': nutrition
                })

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    return rice_features, image, thresh, gray

@app.route('/analyze_rice', methods=['POST'])
def analyze_rice():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    
    # Save the image temporarily
    image_path = "uploaded_rice_image.jpg"
    file.save(image_path)

    features, original_image, processed_image, gray_image = extract_rice_features(image_path)

    if features:
        results = []
        for idx, feature in enumerate(features):
            nutrition = feature['nutrition']
            results.append({
                'grain': idx + 1,
                'type': nutrition['type'].capitalize(),
                'length_cm': round(feature['length_cm'], 2),
                'width_cm': round(feature['width_cm'], 2),
                'volume_cm3': round(feature['volume_cm3'], 4),
                'weight_g': round(nutrition['weight'], 4),
                'calories': round(nutrition['calories'], 2),
                'protein': round(nutrition['protein'], 2),
                'fat': round(nutrition['fat'], 2),
                'carbs': round(nutrition['carbs'], 2)
            })
        return jsonify({"grains": results})
    else:
        return jsonify({"message": "No rice grains detected."})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5004)
