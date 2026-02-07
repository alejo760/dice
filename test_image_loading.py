#!/usr/bin/env python3
"""
Test script to verify image loading works correctly.
Run this to diagnose JPG rendering issues.
"""

import sys
from pathlib import Path

# Check Python version
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("-" * 50)

# Test imports
print("Testing imports...")
try:
    import numpy as np
    print(f"✅ numpy: {np.__version__}")
except ImportError as e:
    print(f"❌ numpy: {e}")

try:
    import cv2
    print(f"✅ opencv: {cv2.__version__}")
except ImportError as e:
    print(f"❌ opencv: {e}")

try:
    from PIL import Image
    import PIL
    print(f"✅ Pillow: {PIL.__version__}")
except ImportError as e:
    print(f"❌ Pillow: {e}")

try:
    import streamlit as st
    print(f"✅ streamlit: {st.__version__}")
except ImportError as e:
    print(f"❌ streamlit: {e}")

print("-" * 50)

# Test image loading functions
def test_load_image(image_path):
    """Test different methods of loading an image."""
    image_path = Path(image_path)
    print(f"\nTesting: {image_path}")
    print(f"  Exists: {image_path.exists()}")
    
    if not image_path.exists():
        print("  ❌ File not found!")
        return None
    
    # Method 1: PIL
    try:
        with Image.open(image_path) as pil_img:
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            img = np.array(pil_img)
            print(f"  ✅ PIL loaded: shape={img.shape}, dtype={img.dtype}")
    except Exception as e:
        print(f"  ❌ PIL failed: {e}")
    
    # Method 2: cv2.imread
    try:
        img = cv2.imread(str(image_path))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"  ✅ cv2.imread loaded: shape={img_rgb.shape}, dtype={img_rgb.dtype}")
        else:
            print(f"  ❌ cv2.imread returned None")
    except Exception as e:
        print(f"  ❌ cv2.imread failed: {e}")
    
    # Method 3: cv2.imdecode (bytes)
    try:
        with open(image_path, 'rb') as f:
            file_bytes = f.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"  ✅ cv2.imdecode loaded: shape={img_rgb.shape}, dtype={img_rgb.dtype}")
        else:
            print(f"  ❌ cv2.imdecode returned None")
    except Exception as e:
        print(f"  ❌ cv2.imdecode failed: {e}")
    
    return img

# Find and test images
print("\nSearching for test images...")
base_dirs = [
    Path("./uploaded_images"),
    Path("./Pacientes"),
    Path("."),
]

test_images = []
for base in base_dirs:
    if base.exists():
        for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.png"]:
            test_images.extend(list(base.glob(f"**/{ext}"))[:2])

if test_images:
    print(f"Found {len(test_images)} images to test")
    for img_path in test_images[:5]:  # Test up to 5 images
        test_load_image(img_path)
else:
    print("No images found. Creating a test image...")
    # Create a simple test image
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_path = Path("./test_image.jpg")
    cv2.imwrite(str(test_path), test_img)
    print(f"Created test image: {test_path}")
    test_load_image(test_path)

print("\n" + "=" * 50)
print("Test complete!")
print("=" * 50)
