import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Function to classify grains based on aspect ratio
def get_classification(ratio):
    ratio = round(ratio, 2)
    if ratio > 1.5:
        return "Full"
    elif 1.2 <= ratio <= 1.5:
        return "Medium"
    else:
        return "Broken"

# Function to calculate nutritional values
def get_nutritional_values(full_count, medium_count, broken_count):
    # Average weight per grain (in grams)
    avg_weight_per_full_grain = 0.03
    avg_weight_per_medium_grain = 0.025
    avg_weight_per_broken_grain = 0.02

    # Updated nutritional values per 100 grams of rice
    nutrition_per_100g = {
        "Calories": 350,
        "Protein (g)": 6.8,
        "Carbs (g)": 78.0,
        "Fat (g)": 0.5
    }

    # Total weight of all grains
    total_weight = (
        full_count * avg_weight_per_full_grain
        + medium_count * avg_weight_per_medium_grain
        + broken_count * avg_weight_per_broken_grain
    )

    # Calculate nutritional values proportionally
    nutrition = {k: round(v * total_weight / 100, 2) for k, v in nutrition_per_100g.items()}

    return total_weight, nutrition

# Route to receive the image and process it
@app.route('/classify_grains', methods=['POST'])
def classify_grains():
    # Ensure the request has an image file
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    
    # Save the image temporarily
    img_path = "uploaded_image.jpg"
    file.save(img_path)

    # Load image in grayscale mode
    img = cv2.imread(img_path, 0)

    if img is None:
        return jsonify({"error": "Failed to load image"}), 400

    # Convert to binary
    _, binary = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)

    # Apply an averaging filter
    kernel = np.ones((5, 5), np.float32) / 9
    filtered = cv2.filter2D(binary, -1, kernel)

    # Morphological operations (erosion and dilation)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erosion = cv2.erode(filtered, kernel2, iterations=1)
    dilation = cv2.dilate(erosion, kernel2, iterations=1)

    # Find contours for grain detection
    contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    full_count = 0
    medium_count = 0
    broken_count = 0
    total_aspect_ratio = 0

    # Classify each grain and count them
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if aspect_ratio < 1:
            aspect_ratio = 1 / aspect_ratio

        classification = get_classification(aspect_ratio)
        total_aspect_ratio += aspect_ratio
        if classification == "Full":
            full_count += 1
        elif classification == "Medium":
            medium_count += 1
        else:
            broken_count += 1

    # Calculate nutritional values
    total_weight, nutrition = get_nutritional_values(full_count, medium_count, broken_count)

    # Prepare the response data in a user-friendly way
    response_data = {
        "summary": {
            "total_grains": full_count + medium_count + broken_count,
            "full_grains": full_count,
            "medium_grains": medium_count,
            "broken_grains": broken_count,
            "total_weight": f"{round(total_weight, 2)} grams"
        },
        "nutritional_values": {
            "calories": f"{round(nutrition['Calories'], 2)} kcal",
            "protein": f"{round(nutrition['Protein (g)'], 2)} g",
            "carbohydrates": f"{round(nutrition['Carbs (g)'], 2)} g",
            "fat": f"{round(nutrition['Fat (g)'], 2)} g"
        }
    }

    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)
