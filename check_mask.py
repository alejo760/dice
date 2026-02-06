#!/usr/bin/env python3
"""Analyze saved annotation mask for area calculation verification."""

import cv2
import numpy as np
from pathlib import Path

mask_path = Path("/Users/alejo/Library/CloudStorage/OneDrive-HospitalAlmaMáter/Validación Humath/data/Pacientes/7035909/7035909_20240326_mask.png")
orig_path = Path("/Users/alejo/Library/CloudStorage/OneDrive-HospitalAlmaMáter/Validación Humath/data/Pacientes/7035909/7035909_20240326.jpg")

# Load mask
mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
orig = cv2.imread(str(orig_path))

print("=" * 50)
print("MASK ANALYSIS")
print("=" * 50)
print(f"Mask shape: {mask.shape}")
print(f"Mask dtype: {mask.dtype}")
print(f"Mask min: {mask.min()}, max: {mask.max()}")
print(f"Unique values (first 10): {np.unique(mask)[:10]}")

print("\n" + "=" * 50)
print("ORIGINAL IMAGE")
print("=" * 50)
print(f"Original shape: {orig.shape}")
print(f"Dimensions match: {mask.shape[:2] == orig.shape[:2]}")

print("\n" + "=" * 50)
print("AREA STATISTICS")
print("=" * 50)
total_pixels = mask.shape[0] * mask.shape[1]
annotated_pixels = int(np.sum(mask > 0))
area_percent = (annotated_pixels / total_pixels) * 100

print(f"Image dimensions: {mask.shape[1]} x {mask.shape[0]} px")
print(f"Total pixels: {total_pixels:,}")
print(f"Annotated pixels: {annotated_pixels:,}")
print(f"Annotated area: {area_percent:.2f}%")

print("\n" + "=" * 50)
print("VERDICT")
print("=" * 50)
if mask.shape[:2] == orig.shape[:2] and annotated_pixels > 0:
    print("✅ Mask is CORRECT for area calculation!")
    print("   - Resized to original image dimensions")
    print("   - Contains valid annotation data")
else:
    print("❌ Issues found:")
    if mask.shape[:2] != orig.shape[:2]:
        print("   - Mask dimensions don't match original")
    if annotated_pixels == 0:
        print("   - No annotated pixels found")
