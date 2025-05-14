from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os
import uuid
import base64
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create a temporary directory for storing images if it doesn't exist
UPLOAD_FOLDER = 'temp_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to classify grain type based on aspect ratio
def get_classification(ratio):
    ratio = round(ratio, 1)
    if ratio >= 1.5:
        return "Long Grain"
    elif 1.0 <= ratio < 1.5:
        return "Medium Grain"
    else:
        return "Short Grain"

# Function to estimate nutritional values based on grain type
def get_nutritional_values(grain_type, grain_count):
    # Nutritional values per 100 grams (average)
    nutrition_per_100g = {
        "Short Grain": {"Calories": 360, "Protein (g)": 6.5, "Carbs (g)": 79, "Fat (g)": 0.5},
        "Medium Grain": {"Calories": 358, "Protein (g)": 7.0, "Carbs (g)": 78, "Fat (g)": 0.4},
        "Long Grain": {"Calories": 365, "Protein (g)": 7.5, "Carbs (g)": 77, "Fat (g)": 0.3}
    }

    # Average weight per grain (in grams)
    avg_weight_per_grain = {
        "Short Grain": 0.020,
        "Medium Grain": 0.025,
        "Long Grain": 0.030
    }

    # Calculate total grain weight
    total_weight = grain_count * avg_weight_per_grain.get(grain_type, 0.025)

    # Calculate nutritional values based on weight
    nutrition = {k: round(v * total_weight / 100, 2) for k, v in nutrition_per_100g[grain_type].items()}

    return total_weight, nutrition

# Function to analyze rice grains
def analyze_rice_image(image_path):
    # Load the image
    img = cv2.imread(image_path, 0)  # grayscale
    original_color = cv2.imread(image_path)  # color

    if img is None:
        return {"error": "Could not process image"}, None, None
    
    # Convert to binary
    _, binary = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)

    # Averaging filter
    kernel = np.ones((5, 5), np.float32) / 25
    filtered = cv2.filter2D(binary, -1, kernel)

    # Morphological operations
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erosion = cv2.erode(filtered, kernel2, iterations=1)
    dilation = cv2.dilate(erosion, kernel2, iterations=1)

    # Size detection
    contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grain_count = len(contours)
    
    # Create a copy of the original image for outlining rice grains
    outlined_rice = original_color.copy()
    
    # Draw all contours on the outlined image with different colors
    cv2.drawContours(outlined_rice, contours, -1, (0, 255, 0), 2)
    
    # Process aspect ratios
    total_aspect_ratio = 0
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if aspect_ratio < 1:
            aspect_ratio = 1 / aspect_ratio
        total_aspect_ratio += aspect_ratio
        
        # Draw bounding rectangle
        color = (0, 0, 255)  # Red color for bounding box
        cv2.rectangle(outlined_rice, (x, y), (x+w, y+h), color, 2)
        
        # Classify individual grain
        grain_type = get_classification(aspect_ratio)
        
        # Add grain ID and type to the image (only for the first 10 grains to avoid cluttering)
        if i < 10:
            cv2.putText(outlined_rice, f"#{i+1}: {grain_type}", (x, y-5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Calculate average aspect ratio and determine grain type
    if grain_count > 0:
        avg_aspect_ratio = total_aspect_ratio / grain_count
        grain_type = get_classification(avg_aspect_ratio)
        
        # Get nutritional values
        total_weight, nutrition = get_nutritional_values(grain_type, grain_count)
        
        # Add summary text to the image
        summary_text = [
            f"Rice Count: {grain_count}",
            f"Type: {grain_type}",
            f"Avg Ratio: {round(avg_aspect_ratio, 2)}",
            f"Est Weight: {round(total_weight, 2)}g"
        ]
        
        # Add summary to bottom of image
        for i, line in enumerate(summary_text):
            cv2.putText(outlined_rice, line, (10, outlined_rice.shape[0] - 10 - (i * 20)), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(outlined_rice, line, (10, outlined_rice.shape[0] - 10 - (i * 20)), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Prepare results
        results = {
            "grain_count": grain_count,
            "grain_type": grain_type,
            "aspect_ratio": round(avg_aspect_ratio, 2),
            "total_weight": round(total_weight, 2),
            "nutrition": nutrition
        }
        
        return results, outlined_rice, original_color
    else:
        return {"error": "No grains detected"}, None, None

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    # Save the uploaded file temporarily
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        # Analyze the image
        results, outlined_image, _ = analyze_rice_image(filepath)
        
        if "error" in results:
            return jsonify(results), 400
        
        # Save the outlined image
        outlined_filename = 'outlined_' + filename
        outlined_filepath = os.path.join(UPLOAD_FOLDER, outlined_filename)
        cv2.imwrite(outlined_filepath, outlined_image)
        
        # Convert the image to Base64 to send to the client
        _, buffer = cv2.imencode('.jpg', outlined_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Add the image data to the results
        results["outlined_image"] = img_base64
        
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up - remove the temporary files
        if os.path.exists(filepath):
            os.remove(filepath)
        if 'outlined_filepath' in locals() and os.path.exists(outlined_filepath):
            os.remove(outlined_filepath)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)