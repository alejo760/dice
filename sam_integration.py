"""
SAM (Segment Anything Model) Integration for Pneumonia Consolidation
This script uses Meta's Segment Anything Model to generate initial segmentation masks
that can be refined manually.
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import argparse


def setup_sam():
    """
    Setup SAM model. Install with:
    pip install segment-anything
    
    Download checkpoint from:
    https://github.com/facebookresearch/segment-anything#model-checkpoints
    """
    try:
        from segment_anything import sam_model_registry, SamPredictor
        return sam_model_registry, SamPredictor
    except ImportError:
        print("Error: segment-anything not installed.")
        print("Install with: pip install segment-anything")
        print("Then download a model checkpoint from:")
        print("https://github.com/facebookresearch/segment-anything#model-checkpoints")
        return None, None


def initialize_sam_predictor(checkpoint_path, model_type="vit_h"):
    """
    Initialize SAM predictor.
    
    Args:
        checkpoint_path: Path to SAM checkpoint (.pth file)
        model_type: Model type ('vit_h', 'vit_l', or 'vit_b')
    
    Returns:
        SAM predictor object
    """
    sam_model_registry, SamPredictor = setup_sam()
    if sam_model_registry is None:
        return None
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    predictor = SamPredictor(sam)
    
    return predictor


def predict_consolidation_with_points(image_path, predictor, point_coords, point_labels):
    """
    Generate segmentation mask using point prompts.
    
    Args:
        image_path: Path to chest X-ray image
        predictor: SAM predictor object
        point_coords: Array of [x, y] coordinates for prompts
        point_labels: Array of labels (1 for positive/include, 0 for negative/exclude)
    
    Returns:
        mask: Binary segmentation mask
        scores: Confidence scores for each mask
    """
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Set image in predictor
    predictor.set_image(image)
    
    # Convert points to numpy array
    point_coords = np.array(point_coords)
    point_labels = np.array(point_labels)
    
    # Predict
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    
    return masks, scores, image


def predict_consolidation_with_box(image_path, predictor, box_coords):
    """
    Generate segmentation mask using bounding box prompt.
    
    Args:
        image_path: Path to chest X-ray image
        predictor: SAM predictor object
        box_coords: [x1, y1, x2, y2] bounding box coordinates
    
    Returns:
        mask: Binary segmentation mask
    """
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Set image in predictor
    predictor.set_image(image)
    
    # Convert box to numpy array
    box = np.array(box_coords)
    
    # Predict
    masks, scores, logits = predictor.predict(
        box=box,
        multimask_output=True
    )
    
    return masks, scores, image


def automatic_consolidation_detection(image_path, predictor, grid_size=5):
    """
    Automatically detect potential consolidation regions using grid-based sampling.
    
    Args:
        image_path: Path to chest X-ray image
        predictor: SAM predictor object
        grid_size: Number of points in grid (grid_size x grid_size)
    
    Returns:
        Combined mask from multiple detections
    """
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Set image in predictor
    predictor.set_image(image)
    
    # Create grid of points in lung region (avoid edges)
    margin_h = int(h * 0.1)
    margin_w = int(w * 0.2)
    
    x_coords = np.linspace(margin_w, w - margin_w, grid_size)
    y_coords = np.linspace(margin_h, h - margin_h, grid_size)
    
    all_masks = []
    
    for x in x_coords:
        for y in y_coords:
            point = np.array([[x, y]])
            label = np.array([1])
            
            try:
                masks, scores, _ = predictor.predict(
                    point_coords=point,
                    point_labels=label,
                    multimask_output=False
                )
                
                # Only keep masks with high confidence
                if scores[0] > 0.8:
                    all_masks.append(masks[0])
            except Exception as e:
                continue
    
    if not all_masks:
        return None, image
    
    # Combine masks
    combined_mask = np.any(all_masks, axis=0).astype(np.uint8)
    
    return combined_mask, image


def visualize_sam_results(image, masks, scores, point_coords=None, save_path=None):
    """
    Visualize SAM segmentation results.
    
    Args:
        image: Original image
        masks: Array of masks
        scores: Confidence scores
        point_coords: Optional point prompts to display
        save_path: Optional path to save visualization
    """
    fig, axes = plt.subplots(1, len(masks) + 1, figsize=(15, 5))
    
    # Show original image
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    if point_coords is not None:
        axes[0].scatter(point_coords[:, 0], point_coords[:, 1], 
                       c='red', s=100, marker='*')
    
    # Show each mask
    for idx, (mask, score) in enumerate(zip(masks, scores)):
        axes[idx + 1].imshow(image)
        axes[idx + 1].imshow(mask, alpha=0.5, cmap='jet')
        axes[idx + 1].set_title(f'Mask {idx + 1}\nScore: {score:.3f}')
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def save_mask(mask, output_path):
    """Save binary mask as image."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    cv2.imwrite(str(output_path), mask_uint8)
    print(f"Mask saved to: {output_path}")


def interactive_sam_segmentation(image_path, checkpoint_path):
    """
    Interactive segmentation where user clicks points to guide SAM.
    This is a simple CLI version - for GUI, integrate with Streamlit.
    """
    print("Initializing SAM...")
    predictor = initialize_sam_predictor(checkpoint_path)
    
    if predictor is None:
        return
    
    # Load and display image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("\nInstructions:")
    print("1. The image will be displayed")
    print("2. Click on consolidation areas (left click)")
    print("3. Click on background areas to exclude (right click)")
    print("4. Press 'q' when done")
    print("5. Choose best mask from results")
    
    point_coords = []
    point_labels = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point_coords.append([x, y])
            point_labels.append(1)
            print(f"Added positive point at ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            point_coords.append([x, y])
            point_labels.append(0)
            print(f"Added negative point at ({x}, {y})")
    
    # Convert for OpenCV display
    display_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback)
    
    while True:
        display = display_img.copy()
        
        # Draw points
        for coord, label in zip(point_coords, point_labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(display, tuple(coord), 5, color, -1)
        
        cv2.imshow('Image', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    if point_coords:
        print("\nGenerating masks...")
        masks, scores, _ = predict_consolidation_with_points(
            image_path, predictor, point_coords, point_labels
        )
        
        # Visualize results
        visualize_sam_results(image, masks, scores, np.array(point_coords))
        
        # Save best mask
        best_idx = np.argmax(scores)
        output_path = Path(image_path).parent / f"{Path(image_path).stem}_sam_mask.png"
        save_mask(masks[best_idx], output_path)
        
        return masks[best_idx]
    
    return None


def batch_process_with_sam(input_dir, output_dir, checkpoint_path, mode='auto'):
    """
    Batch process images with SAM.
    
    Args:
        input_dir: Directory with chest X-ray images
        output_dir: Directory to save masks
        checkpoint_path: Path to SAM checkpoint
        mode: 'auto' for automatic or 'center' for single center point
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Initializing SAM...")
    predictor = initialize_sam_predictor(checkpoint_path)
    
    if predictor is None:
        return
    
    images = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    print(f"Found {len(images)} images to process")
    
    for img_path in images:
        print(f"\nProcessing: {img_path.name}")
        
        try:
            if mode == 'auto':
                mask, image = automatic_consolidation_detection(img_path, predictor)
            else:
                # Use center point as prompt
                image = cv2.imread(str(img_path))
                h, w = image.shape[:2]
                center_point = [[w // 2, h // 2]]
                masks, scores, image = predict_consolidation_with_points(
                    img_path, predictor, center_point, [1]
                )
                mask = masks[np.argmax(scores)]
            
            if mask is not None:
                output_file = output_path / f"{img_path.stem}_mask.png"
                save_mask(mask, output_file)
            else:
                print(f"No mask generated for {img_path.name}")
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    
    print(f"\nBatch processing complete! Masks saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate pneumonia consolidation masks using SAM"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to SAM checkpoint file (.pth)'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to single image (for interactive mode)'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Input directory for batch processing'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory for batch processing'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='interactive',
        choices=['interactive', 'auto', 'center'],
        help='Processing mode'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='vit_h',
        choices=['vit_h', 'vit_l', 'vit_b'],
        help='SAM model type'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'interactive' and args.image:
        interactive_sam_segmentation(args.image, args.checkpoint)
    elif args.input_dir and args.output_dir:
        batch_process_with_sam(args.input_dir, args.output_dir, 
                              args.checkpoint, args.mode)
    else:
        parser.print_help()
