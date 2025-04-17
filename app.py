from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

app = Flask(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
def load_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(in_features=512, out_features=25)  # Adjust based on your model
    model.load_state_dict(torch.load("model_25.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Labels
labels_for_viz = {
    0: 'Corn__Blight', 1: 'Corn__Common_Rust', 2: 'Corn__Gray_Leaf_Spot', 3: 'Corn__Healthy',
    4: 'Mango__Anthracnose', 5: 'Mango__Bacterial_Canker', 6: 'Mango__Cutting_Weevil',
    7: 'Mango__Die_Back', 8: 'Mango__Gall_Midge', 9: 'Mango__Healthy',
    10: 'Mango__Powdery_Mildew', 11: 'Mango__Sooty_Mould', 12: 'Potato__EarlyBlight',
    13: 'Potato__Healthy', 14: 'Potato__LateBlight', 15: 'Tomato__Bacterial_spot',
    16: 'Tomato__Early_blight', 17: 'Tomato__Late_blight', 18: 'Tomato__Leaf_Mold',
    19: 'Tomato__Septoria_leaf_spot', 20: 'Tomato__Spider_mites Two-spotted_spider_mite',
    21: 'Tomato__Target_Spot', 22: 'Tomato__Tomato_Yellow_Leaf_Curl_Virus',
    23: 'Tomato__Tomato_mosaic_virus', 24: 'Tomato__healthy'
}

# Image Preprocessing
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor_image = preprocess(image).unsqueeze(0).to(device)
    return tensor_image

# Prediction Function
def classify_disease(image):
    tensor_image = preprocess_image(image)
    with torch.no_grad():
        output = model(tensor_image)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()
    return labels_for_viz[prediction]

# API Route for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    try:
        image = Image.open(file)  # Convert image to RGB
       
        prediction = classify_disease( image)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Running on port 5500 and allowing external access
    app.run(host='0.0.0.0', port=5500, debug=True)




