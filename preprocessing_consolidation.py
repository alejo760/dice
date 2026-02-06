"""
Preprocessing script for pneumonia consolidation enhancement.
This script enhances chest X-ray images to better visualize consolidations,
air bronchograms, and subtle patterns necessary for accurate segmentation.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def enhance_consolidation(img_path, output_path=None):
    """
    Enhance chest X-ray image to better visualize pneumonia consolidation.
    
    Args:
        img_path: Path to the input image (JPG/PNG)
        output_path: Path to save the enhanced image (optional)
        
    Returns:
        Enhanced image as numpy array
    """
    # Read image in grayscale
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Could not read image from {img_path}")
    
    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Enhances local contrast to see subtle consolidations
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    
    # 2. Sharpening filter to reveal air bronchograms
    # Air bronchograms are dark branch-like patterns inside consolidation
    kernel = np.array([[-1, -1, -1], 
                       [-1,  9, -1], 
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # 3. Optional: Edge enhancement to see consolidation boundaries
    # Using Laplacian for edge detection
    laplacian = cv2.Laplacian(sharpened, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Combine sharpened image with edge information
    alpha = 0.8  # Weight for sharpened image
    beta = 0.2   # Weight for edge information
    result = cv2.addWeighted(sharpened, alpha, laplacian, beta, 0)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(str(output_path), result)
        print(f"Enhanced image saved to: {output_path}")
    
    return result


def batch_enhance_consolidation(input_dir, output_dir, image_extension='.jpg'):
    """
    Process multiple images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save enhanced images
        image_extension: File extension to process (default: .jpg)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images with specified extension
    images = list(input_path.rglob(f"*{image_extension}"))
    
    print(f"Found {len(images)} images to process")
    
    for img_path in images:
        try:
            # Create output path maintaining relative structure
            rel_path = img_path.relative_to(input_path)
            out_path = output_path / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Process image
            enhance_consolidation(img_path, out_path)
            print(f"Processed: {img_path.name}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"\nProcessing complete! Enhanced images saved to: {output_path}")


def create_visualization_comparison(original_path, enhanced_path, output_path):
    """
    Create a side-by-side comparison of original and enhanced images.
    
    Args:
        original_path: Path to original image
        enhanced_path: Path to enhanced image
        output_path: Path to save comparison image
    """
    original = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
    enhanced = cv2.imread(str(enhanced_path), cv2.IMREAD_GRAYSCALE)
    
    # Resize if needed to match dimensions
    if original.shape != enhanced.shape:
        enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
    
    # Create side-by-side comparison
    comparison = np.hstack([original, enhanced])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, 'Enhanced', (original.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
    
    cv2.imwrite(str(output_path), comparison)
    print(f"Comparison saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhance chest X-ray images for pneumonia consolidation segmentation"
    )
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Input image file or directory'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help='Output file or directory'
    )
    parser.add_argument(
        '--batch', 
        action='store_true',
        help='Process entire directory (batch mode)'
    )
    parser.add_argument(
        '--extension', 
        type=str, 
        default='.jpg',
        help='Image file extension for batch processing (default: .jpg)'
    )
    
    args = parser.parse_args()
    
    if args.batch:
        batch_enhance_consolidation(args.input, args.output, args.extension)
    else:
        enhance_consolidation(args.input, args.output)
