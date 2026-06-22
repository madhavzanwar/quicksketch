# backend/debug_image.py

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def debug_preprocess(image_path):

    # Load image
    img = Image.open(image_path)

    print(f"Original mode: {img.mode}")
    print(f"Original size: {img.size}")

    # Fix RGBA transparency before grayscale conversion
    if img.mode in ('RGBA', 'LA') or \
       (img.mode == 'P' and 'transparency' in img.info):

        background = Image.new('RGB', img.size, (255, 255, 255))

        if img.mode == 'P':
            img = img.convert('RGBA')

        background.paste(img, mask=img.split()[-1])
        img = background

    # Step 1: Convert to grayscale
    img_gray = img.convert('L')

    # Step 2: Resize to 28x28
    img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)

    # Step 3: Convert to numpy
    arr = np.array(img_resized, dtype=np.float32)

    # Step 4: Normalize
    arr_normalized = arr / 255.0

    # Step 5: Invert colors
    arr_inverted = 1.0 - arr_normalized

    # Plot all stages
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle('What your model actually sees', fontsize=13)

    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('1. Grayscale')
    axes[0].axis('off')

    axes[1].imshow(img_resized, cmap='gray')
    axes[1].set_title('2. Resized 28x28')
    axes[1].axis('off')

    axes[2].imshow(arr_normalized, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('3. Normalized')
    axes[2].axis('off')

    axes[3].imshow(arr_inverted, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title('4. Inverted (what model sees)')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig('debug_output.png', dpi=150)
    plt.show()

    print(f"\nAfter normalization - min: {arr_normalized.min():.3f}, max: {arr_normalized.max():.3f}")
    print(f"After inversion   - min: {arr_inverted.min():.3f}, max: {arr_inverted.max():.3f}")

    stroke_pixels_inverted = (arr_inverted > 0.5).sum()
    stroke_pixels_normal = (arr_normalized > 0.5).sum()

    print(f"\nPixels > 0.5 (stroke pixels):")
    print(f"  Without inversion: {stroke_pixels_normal}")
    print(f"  With inversion:    {stroke_pixels_inverted}")

    print(f"\nQuick Draw format needs MORE dark background than stroke pixels")
    print(f"Correct format: stroke pixels should be minority of total {28*28} pixels")


if __name__ == '__main__':
    image_path = os.path.join(os.path.dirname(__file__), 'test_sketch.png')
    debug_preprocess(image_path)