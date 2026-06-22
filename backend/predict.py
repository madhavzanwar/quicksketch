# backend/predict.py

import torch
import numpy as np
from PIL import Image
import io
import base64
import os
from models.cnn import QuickDrawCNN
from utils.dataset import CATEGORIES

DEBUG_SAVE = os.environ.get('QUICKSKETCH_DEBUG', '1') == '1'
DEBUG_DIR = os.path.dirname(__file__)

STROKE_THRESHOLD = 250
PADDING_RATIO = 0.15
OUTPUT_SIZE = 28

DEBUG_STEPS = {
    'decoded': 'debug_1_decoded.png',
    'gray': 'debug_2_gray.png',
    'crop': 'debug_3_crop.png',
    'resize': 'debug_4_resize.png',
    'invert': 'debug_5_invert.png',
    'final': 'debug_6_final.png',
}


def load_model(model_path):
    """Load the trained model from a .pth checkpoint file."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = QuickDrawCNN(num_classes=len(CATEGORIES))
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    print(f"Model loaded from: {model_path}")
    print(f"Trained for {checkpoint['epoch'] + 1} epochs")
    print(f"Best validation accuracy: {checkpoint['val_acc']:.1f}%")

    return model, device


def _save_debug_step(img, step_key, save_debug):
    """Save a PIL image or numpy array for a named debug step."""
    if not save_debug:
        return

    path = os.path.join(DEBUG_DIR, DEBUG_STEPS[step_key])
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img, mode='L').save(path)
    else:
        img.save(path)

    arr = np.array(img if not isinstance(img, np.ndarray) else img)
    print(f"  [{step_key}] saved {path} — shape={arr.shape}, min={arr.min()}, max={arr.max()}")


def _decode_image(image_data):
    """
    Decode base64 string, file path, or bytes into a PIL Image.

    Important: canvas base64 is long and may contain '/' characters.
    Never call os.path.exists() on base64 — it can be misread as a path on Windows.
    """
    if isinstance(image_data, bytes):
        return Image.open(io.BytesIO(image_data))

    if isinstance(image_data, str):
        if image_data.startswith('data:image'):
            image_data = image_data.split(',', 1)[1]

        # Only treat as a file path when it is clearly a short local path
        is_likely_base64 = len(image_data) > 512 or '/' in image_data[:20] or image_data.startswith('iVBOR')
        if not is_likely_base64 and os.path.isfile(image_data):
            return Image.open(image_data)

        image_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(image_bytes))

    raise ValueError("image_data must be a file path, base64 string, or bytes")


def _flatten_alpha(img):
    """Paste RGBA/transparency onto white background."""
    if img.mode in ('RGBA', 'LA') or \
       (img.mode == 'P' and 'transparency' in img.info):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1])
        return background
    return img


def _find_stroke_bbox(arr):
    """Return bounding box (rmin, rmax, cmin, cmax) of dark stroke pixels."""
    stroke_mask = arr < STROKE_THRESHOLD
    if not stroke_mask.any():
        return None

    rows = np.any(stroke_mask, axis=1)
    cols = np.any(stroke_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    pad = int(max(rmax - rmin + 1, cmax - cmin + 1) * PADDING_RATIO)
    rmin = max(0, rmin - pad)
    rmax = min(arr.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(arr.shape[1] - 1, cmax + pad)

    return rmin, rmax, cmin, cmax


def _crop_to_square(arr, bbox):
    """Crop array to bounding box and center in a white square."""
    rmin, rmax, cmin, cmax = bbox
    cropped = arr[rmin:rmax + 1, cmin:cmax + 1]
    ch, cw = cropped.shape
    side = max(ch, cw)

    square = np.full((side, side), 255, dtype=np.uint8)
    y_off = (side - ch) // 2
    x_off = (side - cw) // 2
    square[y_off:y_off + ch, x_off:x_off + cw] = cropped

    return cropped, square


def _dilate(binary, radius=1):
    """Expand stroke mask by `radius` pixels using max-filter dilation."""
    result = binary.copy()
    h, w = binary.shape
    for _ in range(radius):
        padded = np.pad(result, 1, mode='constant', constant_values=0)
        expanded = result.copy()
        for dy in range(3):
            for dx in range(3):
                expanded = np.maximum(
                    expanded,
                    padded[dy:dy + h, dx:dx + w]
                )
        result = expanded
    return result


def preprocess_image(image_data, save_debug=DEBUG_SAVE):
    """
    Convert canvas image into a tensor matching training format.

    Training (dataset.py): QuickDraw .npy → float32 / 255 → (1, 28, 28)
    QuickDraw format: white strokes (1.0) on black background (0.0).

    Pipeline:
      decode → grayscale → crop bbox → square → binarize → resize (NEAREST) → dilate
    """
    # Step 1: decode
    img = _decode_image(image_data)
    _save_debug_step(img.convert('RGB') if img.mode == 'RGBA' else img, 'decoded', save_debug)

    img = _flatten_alpha(img)

    # Step 2: grayscale (black strokes on white background)
    img_gray = img.convert('L')
    _save_debug_step(img_gray, 'gray', save_debug)

    arr = np.array(img_gray, dtype=np.uint8)
    gray_strokes = int((arr < STROKE_THRESHOLD).sum())
    print(f"  Grayscale stroke pixels (<{STROKE_THRESHOLD}): {gray_strokes} / {arr.size}")

    bbox = _find_stroke_bbox(arr)
    if bbox is None:
        print("  WARNING: No stroke pixels found in grayscale — returning empty tensor")
        empty = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
        _save_debug_step(empty, 'crop', save_debug)
        _save_debug_step(empty, 'resize', save_debug)
        _save_debug_step(empty, 'invert', save_debug)
        _save_debug_step(empty, 'final', save_debug)
        return torch.tensor(empty).unsqueeze(0).unsqueeze(0)

    # Step 3: crop to drawing bounding box
    cropped, square = _crop_to_square(arr, bbox)
    _save_debug_step(Image.fromarray(cropped, mode='L'), 'crop', save_debug)

    # Step 4: binarize BEFORE resize (avoids LANCZOS washing out thin strokes)
    # Dark pixels → 255 (stroke), light pixels → 0 (background)
    binary_square = np.where(square < STROKE_THRESHOLD, 255, 0).astype(np.uint8)
    resized_binary = Image.fromarray(binary_square, mode='L').resize(
        (OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.NEAREST
    )
    _save_debug_step(resized_binary, 'resize', save_debug)

    # Step 5: invert to QuickDraw format — white strokes (1.0) on black (0.0)
    stroke = (np.array(resized_binary, dtype=np.float32) / 255.0)
    stroke = _dilate(stroke, radius=1)
    inverted_vis = (stroke * 255).astype(np.uint8)
    _save_debug_step(inverted_vis, 'invert', save_debug)

    # Step 6: final tensor input
    img_array = stroke
    _save_debug_step(inverted_vis, 'final', save_debug)

    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)

    print(f"  Preprocessed tensor shape: {tuple(img_tensor.shape)}")
    print(f"  Pixel range (model input): min={img_array.min():.3f}, max={img_array.max():.3f}")
    print(f"  Stroke pixels (>0.5): {(img_array > 0.5).sum()} / {OUTPUT_SIZE * OUTPUT_SIZE}")

    return img_tensor


def predict(model, image_data, device, top_k=3):
    """Run inference and return top-k predictions."""
    img_tensor = preprocess_image(image_data)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

    top_probs = top_probs.squeeze().cpu().numpy()
    top_indices = top_indices.squeeze().cpu().numpy()

    if top_k == 1:
        top_probs = np.array([top_probs])
        top_indices = np.array([top_indices])

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            'category': CATEGORIES[int(idx)],
            'confidence': float(round(prob * 100, 2)),
            'label_idx': int(idx)
        })

    return results


if __name__ == '__main__':
    import sys

    model_path = os.path.join(
        os.path.dirname(__file__),
        'models',
        'best_model.pth'
    )

    if not os.path.exists(model_path):
        print("ERROR: No trained model found.")
        print(f"Expected at: {model_path}")
        print("Run train.py first.")
        sys.exit(1)

    model, device = load_model(model_path)

    print("\nTesting with random noise image...")
    fake_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    fake_pil = Image.fromarray(fake_image, mode='L')

    temp_path = os.path.join(os.path.dirname(__file__), 'temp_test.png')
    fake_pil.save(temp_path)

    results = predict(model, temp_path, device, top_k=3)

    print("\nTop 3 predictions (random noise - results meaningless):")
    print("-" * 40)
    for i, result in enumerate(results):
        print(f"{i+1}. {result['category']:12s} {result['confidence']:.2f}%")

    os.remove(temp_path)
    print("\nInference pipeline working correctly.")
