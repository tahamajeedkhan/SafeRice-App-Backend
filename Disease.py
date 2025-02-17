from flask import Flask, request, jsonify
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2
from flask_cors import CORS  # CORS for cross-origin requests

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load the trained PyTorch model
model_path = 'E:\\Trained Models\\disease_model_full.pth' 
try:
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()  # Set the model to evaluation mode
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Define your disease classes (replace these with your actual class names)
class_names = [
    'Brown spot',
    'Leaf Blast',
    'Narrow Brown Spot',
    'Leaf Scald',
    'Bacterial Leaf Blight',
    'Tungro',
    'Leaf Smut'
]

# Image preprocessing function to prepare input for the model
def preprocess_image(image):
    try:
        transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Resize to match model input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
        ])
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB
        image = transform(image)  # Apply transformations
        image = image.unsqueeze(0)  # Add batch dimension
        return image
    except Exception as e:
        print(f"Error during image preprocessing: {str(e)}")
        return None

# Function to classify the image using the PyTorch model
def classify_image(image):
    try:
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None
        with torch.no_grad():  # Disable gradient computation for inference
            output = model(processed_image)
        predicted_class = torch.argmax(output, dim=1).item()  # Get the class index
        return predicted_class
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        return None

# Route for disease diagnosis
@app.route('/diagnoseDisease', methods=['POST'])
def diagnose_disease():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    
    # Log the received image filename
    print(f"Received image: {image.filename}")
    
    try:
        # Convert the image to an array and decode it
        img_array = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": "Error processing image"}), 500

    # Classify the image and get the predicted class
    predicted_class_index = classify_image(img)
    if predicted_class_index is None:
        return jsonify({"error": "Error during prediction"}), 500

    predicted_disease = class_names[predicted_class_index]

    return jsonify({"diagnosis": predicted_disease})

if __name__ == '__main__':
    if model is None:
        print("Model failed to load. Exiting application.")
    else:
        app.run(debug=True, host='0.0.0.0', port=5002)  # 0.0.0.0 allows access from other devices on the network
