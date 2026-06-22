# backend/predict.py

import torch
import numpy as np
from PIL import Image
import io
import base64
import os
from models.cnn import QuickDrawCNN
from utils.dataset import CATEGORIES

# --- Load the model once at startup ---
# We do this outside the function so the model
# isn't reloaded on every prediction request

def load_model(model_path):
    """
    Load the trained model from a .pth checkpoint file.
    Returns the model in evaluation mode.
    """
    # Use GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize empty model with same architecture as training
    model = QuickDrawCNN(num_classes=len(CATEGORIES))
    
    # Load the saved weights into the model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Switch to eval mode - disables dropout and batch norm
    # during inference (important - predictions will be wrong without this)
    model.eval()
    model.to(device)
    
    print(f"Model loaded from: {model_path}")
    print(f"Trained for {checkpoint['epoch'] + 1} epochs")
    print(f"Best validation accuracy: {checkpoint['val_acc']:.1f}%")
    
    return model, device


def preprocess_image(image_data):
    """
    Convert raw image data into a tensor the model can process.
    Handles base64 strings, file paths, and bytes.
    """
    
    if isinstance(image_data, str):
        if image_data.startswith('data:image'):
            # Strip the base64 header
            image_data = image_data.split(',')[1]
        
        if os.path.exists(image_data):
            img = Image.open(image_data)
        else:
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes))
    
    elif isinstance(image_data, bytes):
        img = Image.open(io.BytesIO(image_data))
    
    else:
        raise ValueError("image_data must be a file path, base64 string, or bytes")
    
    # Fix RGBA/transparency issue:
    # Paste image onto white background before converting to grayscale
    # This prevents transparent pixels becoming black
    if img.mode in ('RGBA', 'LA') or \
       (img.mode == 'P' and 'transparency' in img.info):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1])  # use alpha as mask
        img = background
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Invert: canvas has black strokes on white background
    # Quick Draw training data has white strokes on black background
    # So we invert to match training format
    img_array = 1.0 - img_array
    
    # Reshape to (1, 1, 28, 28)
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor


def predict(model, image_data, device, top_k=3):
    """
    Run inference on an image and return top-k predictions.
    
    Returns a list of dicts:
    [
        {'category': 'cat', 'confidence': 0.92, 'label_idx': 3},
        {'category': 'dog', 'confidence': 0.05, 'label_idx': 7},
        ...
    ]
    """
    
    # Preprocess the image
    img_tensor = preprocess_image(image_data)
    img_tensor = img_tensor.to(device)
    
    # Run inference without computing gradients
    # (gradients are only needed during training)
    with torch.no_grad():
        outputs = model(img_tensor)  # raw scores (logits)
        
        # Convert raw scores to probabilities using softmax
        # dim=1 means softmax across the class dimension
        probabilities = torch.softmax(outputs, dim=1)
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)
    
    # Convert tensors to Python lists
    top_probs = top_probs.squeeze().cpu().numpy()
    top_indices = top_indices.squeeze().cpu().numpy()
    
    # Build results list
    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            'category': CATEGORIES[idx],
            'confidence': float(round(prob * 100, 2)),  # as percentage
            'label_idx': int(idx)
        })
    
    return results


# --- Test the inference pipeline ---
if __name__ == '__main__':
    import sys
    
    # Path to your saved model
    model_path = os.path.join(
        os.path.dirname(__file__),
        'models',
        'best_model.pth'
    )
    
    # Check model exists
    if not os.path.exists(model_path):
        print("ERROR: No trained model found.")
        print(f"Expected at: {model_path}")
        print("Run train.py first.")
        sys.exit(1)
    
    # Load model
    model, device = load_model(model_path)
    
    # Test with a synthetic image (random noise)
    # Real test will use actual canvas drawings
    print("\nTesting with random noise image...")
    fake_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    fake_pil = Image.fromarray(fake_image, mode='L')
    
    # Save temp file to test file path loading
    temp_path = os.path.join(os.path.dirname(__file__), 'temp_test.png')
    fake_pil.save(temp_path)
    
    results = predict(model, temp_path, device, top_k=3)
    
    print("\nTop 3 predictions (random noise - results meaningless):")
    print("-" * 40)
    for i, result in enumerate(results):
        print(f"{i+1}. {result['category']:12s} {result['confidence']:.2f}%")
    
    # Clean up temp file
    os.remove(temp_path)
    
    print("\nInference pipeline working correctly.")
    print("Ready to connect to Flask API.")