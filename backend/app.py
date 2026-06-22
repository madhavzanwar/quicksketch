# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
import sys

# Add backend directory to path so imports work correctly
sys.path.append(os.path.dirname(__file__))

from predict import load_model, predict
from utils.dataset import CATEGORIES
from train import CONFIG

# --- Initialize Flask app ---
app = Flask(__name__)

# Allow requests from any origin (needed for frontend to talk to backend)
CORS(app)

# --- Load model once at startup ---
# We load it here so it's ready for every request
# Loading on every request would be very slow
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pth')

print("Loading model...")
model, device = load_model(MODEL_PATH)
print("Model ready.\n")


# --- Health check endpoint ---
# Useful to verify the API is running before testing predictions
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'categories': CATEGORIES,
        'num_classes': len(CATEGORIES)
    })


# --- Main prediction endpoint ---
@app.route('/predict', methods=['POST'])
def predict_sketch():
    
    # Check that request contains JSON
    if not request.is_json:
        return jsonify({
            'error': 'Request must be JSON'
        }), 400
    
    data = request.get_json()
    
    # Check that image data is present
    if 'image' not in data:
        return jsonify({
            'error': 'Missing image field in request body'
        }), 400
    
    image_data = data['image']
    
    # Validate it's not empty
    if not image_data or len(image_data) < 10:
        return jsonify({
            'error': 'Image data is empty or too short'
        }), 400
    
    # Run prediction
    # Wrap in try/except to return clean error if something goes wrong
    try:
        print("Prediction request received")
        print(f"Image length: {len(image_data)}")

        results = predict(model, image_data, device, top_k=3)

        print(f"Top prediction: {results[0]['category']} ({results[0]['confidence']:.1f}%)")

        return jsonify({
            'success': True,
            'predictions': results,
            # Also return the top prediction separately for easy access
            'top_prediction': results[0]['category'],
            'top_confidence': results[0]['confidence']
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


# --- Categories endpoint ---
# Frontend can use this to know which categories the model supports
@app.route('/categories', methods=['GET'])
def get_categories():
    return jsonify({
        'categories': CATEGORIES,
        'count': len(CATEGORIES)
    })


@app.route('/model-metadata', methods=['GET'])
def get_model_metadata():
    training_samples = int(
        len(CATEGORIES)
        * CONFIG['max_samples_per_class']
        * CONFIG['train_ratio']
    )

    return jsonify({
        'num_classes': len(CATEGORIES),
        'training_samples': training_samples,
        'trainable_parameters': model.count_parameters(),
    })


# --- Run the app ---
if __name__ == '__main__':
    # debug=True means Flask auto-reloads when you save changes
    # Turn this off in production
    app.run(
        host='0.0.0.0',  # accept requests from any IP
        port=5000,
        debug=True
    )
