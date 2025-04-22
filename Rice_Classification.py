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
model_path = 'E:\\Final Year Project Final\\Trained Models\\new_rice_type_model_full.pth'  # Update this with your actual model path
device = torch.device('cpu')  # Use 'cpu' or 'cuda' depending on your environment
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()  # Set the model to evaluation mode

# Define your disease classes (replace these with your actual class names)
class_names = [
    'Arborio',
    'Ipsala', 
    'Super', 
    'Basmati', 
    'Karacadag', 
    '1508', 
    'Sufaid', 
    'Seela', 
    'Kachi_Kainat', 
    'Kachi', 
    'Ari', 
    'Jasmine'
]

def preprocess_image(image):
    try:
        # Convert PIL image to NumPy array
        image = np.array(image)
        
        # Ensure a new copy of the image to avoid reference issues
        image = np.copy(image)
        
        # Resize the image using OpenCV (which avoids memory reference issues)
        image = cv2.resize(image, (299, 299))
        
        # Convert RGB to BGR (OpenCV uses BGR by default)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Convert BGR to LAB and apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab_image = cv2.merge((l_channel, a_channel, b_channel))
        
        # Convert back to BGR
        bgr_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
        
        # Apply median blur
        bgr_image = cv2.medianBlur(bgr_image, ksize=3)
        
        # Apply gamma correction
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        bgr_image = cv2.LUT(bgr_image, table)
        
        # Convert back to PIL Image and normalize
        image = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        image = np.array(image).astype(np.float32)
        
        # Normalize the image to [-1, 1] range
        image = (image / 127.5) - 1.0
        
        # Ensure correct shape
        if image.shape != (299, 299, 3):
            raise ValueError(f"Image shape is {image.shape}, expected (299, 299, 3)")
        
        # Convert to tensor and permute dimensions (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None



# Function to classify the image using the PyTorch model
def classify_image(image):
    try:
        processed_image = preprocess_image(image)
        if processed_image is None:
            raise ValueError("Image preprocessing failed")
        processed_image = processed_image.unsqueeze(0)  # Add batch dimension
        processed_image = processed_image.to(device)  # Move image to correct device

        with torch.no_grad():  # Disable gradient computation for inference
            output = model(processed_image)
        predicted_class = torch.argmax(output, dim=1).item()  # Get the class index
        return predicted_class
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        return None

# Route for disease diagnosis
@app.route('/diagnoseRiceType', methods=['POST'])
def diagnose_rice_type():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    
    # Log the received image filename
    print(f"Received image: {image.filename}")
    
    try:
        # Convert the image to an array and decode it
        img_array = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image")
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
    app.run(debug=True, host='0.0.0.0', port=5003)  # 0.0.0.0 allows access from other devices on the network
