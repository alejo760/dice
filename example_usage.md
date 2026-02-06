# Example Usage - Pneumonia Consolidation Segmentation

This notebook demonstrates how to use the pneumonia consolidation segmentation tools.

## Setup

```python
import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append('..')

# Import our modules
from preprocessing_consolidation import enhance_consolidation
from dice_calculator_app import (
    calculate_dice_coefficient, 
    calculate_iou,
    calculate_precision_recall,
    create_overlay_visualization
)
```

## 1. Preprocessing Images

### Enhance a single image to see consolidation better

```python
# Path to your chest X-ray
input_image = "../data/Pacientes/7035909/7035909_20240326.jpg"
output_image = "../dice/enhanced_images/7035909_enhanced.jpg"

# Enhance the image
enhanced = enhance_consolidation(input_image, output_image)

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

original = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
axes[0].imshow(original, cmap='gray')
axes[0].set_title('Original X-ray')
axes[0].axis('off')

axes[1].imshow(enhanced, cmap='gray')
axes[1].set_title('Enhanced (CLAHE + Sharpening)')
axes[1].axis('off')

plt.tight_layout()
plt.show()
```

### Batch process multiple images

```python
from preprocessing_consolidation import batch_enhance_consolidation

# Process all patient images
input_dir = "../data/Pacientes/"
output_dir = "../dice/enhanced_images/"

batch_enhance_consolidation(input_dir, output_dir, image_extension='.jpg')
```

## 2. Create Sample Masks for Testing

Let's create some sample masks to demonstrate the Dice calculation.

```python
def create_sample_masks(image_path):
    """Create sample ground truth and prediction masks for demo."""
    
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    # Create ground truth mask (simulated consolidation in lower right lung)
    ground_truth = np.zeros((h, w), dtype=np.uint8)
    center_y, center_x = int(h * 0.6), int(w * 0.7)
    
    # Create irregular shape for consolidation
    for i in range(h):
        for j in range(w):
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            noise = np.random.randn() * 20
            if dist + noise < 80:
                ground_truth[i, j] = 255
    
    # Create predicted mask (similar but slightly different)
    prediction = np.zeros((h, w), dtype=np.uint8)
    center_y_pred = int(h * 0.58)  # Slightly shifted
    center_x_pred = int(w * 0.72)
    
    for i in range(h):
        for j in range(w):
            dist = np.sqrt((i - center_y_pred)**2 + (j - center_x_pred)**2)
            noise = np.random.randn() * 25
            if dist + noise < 75:  # Slightly smaller
                prediction[i, j] = 255
    
    return ground_truth, prediction

# Create sample masks
image_path = "../data/Pacientes/7035909/7035909_20240326.jpg"
gt_mask, pred_mask = create_sample_masks(image_path)

# Save masks
cv2.imwrite("../dice/annotations/ground_truth/sample_gt.png", gt_mask)
cv2.imwrite("../dice/annotations/predictions/sample_pred.png", pred_mask)

print("Sample masks created!")
```

## 3. Calculate Dice Coefficient

```python
# Load masks
ground_truth = cv2.imread("../dice/annotations/ground_truth/sample_gt.png", cv2.IMREAD_GRAYSCALE)
prediction = cv2.imread("../dice/annotations/predictions/sample_pred.png", cv2.IMREAD_GRAYSCALE)

# Calculate metrics
dice = calculate_dice_coefficient(ground_truth, prediction)
iou = calculate_iou(ground_truth, prediction)
precision, recall = calculate_precision_recall(ground_truth, prediction)
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("Segmentation Metrics:")
print(f"  Dice Coefficient: {dice:.4f}")
print(f"  IoU (Jaccard):    {iou:.4f}")
print(f"  Precision:        {precision:.4f}")
print(f"  Recall:           {recall:.4f}")
print(f"  F1 Score:         {f1:.4f}")

# Interpretation
if dice > 0.85:
    quality = "Excellent ✓"
elif dice > 0.70:
    quality = "Good (acceptable for fuzzy borders)"
else:
    quality = "Needs review"
    
print(f"\nQuality Assessment: {quality}")
```

## 4. Visualize Results

```python
# Load original image
original = cv2.imread(image_path)

# Create overlay visualization
overlay = create_overlay_visualization(original, ground_truth, prediction, alpha=0.5)

# Display all views
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original X-ray')
axes[0, 0].axis('off')

axes[0, 1].imshow(ground_truth, cmap='Greens')
axes[0, 1].set_title('Ground Truth Mask')
axes[0, 1].axis('off')

axes[1, 0].imshow(prediction, cmap='Reds')
axes[1, 0].set_title('Predicted Mask')
axes[1, 0].axis('off')

axes[1, 1].imshow(overlay)
axes[1, 1].set_title(f'Overlay (Dice: {dice:.3f})')
axes[1, 1].axis('off')

# Add legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Ground Truth'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Prediction'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='y', markersize=10, label='Overlap')
]
axes[1, 1].legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('../dice/results/example_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualization saved to: dice/results/example_visualization.png")
```

## 5. Batch Calculate Dice Scores

Process multiple mask pairs and generate report.

```python
import pandas as pd
from pathlib import Path

def batch_calculate_dice(gt_dir, pred_dir, results_file):
    """Calculate Dice for all mask pairs in directories."""
    
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    
    results = []
    
    # Find all ground truth masks
    gt_masks = list(gt_dir.glob("*.png")) + list(gt_dir.glob("*.jpg"))
    
    for gt_path in gt_masks:
        # Find corresponding prediction
        pred_path = pred_dir / gt_path.name
        
        if not pred_path.exists():
            print(f"Warning: No prediction found for {gt_path.name}")
            continue
        
        # Load masks
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        
        # Calculate metrics
        dice = calculate_dice_coefficient(gt, pred)
        iou = calculate_iou(gt, pred)
        precision, recall = calculate_precision_recall(gt, pred)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'Image': gt_path.name,
            'Dice': dice,
            'IoU': iou,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })
        
        print(f"Processed: {gt_path.name} - Dice: {dice:.4f}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary = {
        'Metric': ['Mean', 'Std', 'Min', 'Max', 'Median'],
        'Dice': [
            df['Dice'].mean(),
            df['Dice'].std(),
            df['Dice'].min(),
            df['Dice'].max(),
            df['Dice'].median()
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    
    # Save results
    with pd.ExcelWriter(results_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Individual Results', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\nResults saved to: {results_file}")
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))
    
    return df, summary_df

# Run batch processing
gt_directory = "../dice/annotations/ground_truth/"
pred_directory = "../dice/annotations/predictions/"
results_excel = "../dice/results/dice_scores_report.xlsx"

df_results, df_summary = batch_calculate_dice(gt_directory, pred_directory, results_excel)
```

## 6. Working with Real Patient Data

Example of processing actual patient X-rays from your dataset.

```python
# Get list of patient directories
patients_dir = Path("../data/Pacientes/")
patient_folders = [d for d in patients_dir.iterdir() if d.is_dir() and d.name.isdigit()]

print(f"Found {len(patient_folders)} patient folders")

# Process first 5 patients as example
for patient_dir in patient_folders[:5]:
    patient_id = patient_dir.name
    print(f"\nProcessing Patient: {patient_id}")
    
    # Find X-ray image
    images = list(patient_dir.glob("*.jpg"))
    
    if images:
        xray_path = images[0]
        print(f"  X-ray: {xray_path.name}")
        
        # Enhance image
        output_path = f"../dice/enhanced_images/{patient_id}_enhanced.jpg"
        enhanced = enhance_consolidation(str(xray_path), output_path)
        
        print(f"  Enhanced image saved: {output_path}")
        
        # Here you would:
        # 1. Load or create annotations
        # 2. Calculate Dice if annotations exist
        # 3. Generate reports
    else:
        print(f"  No images found")
```

## 7. Quality Control Report

Generate a comprehensive quality control report.

```python
def generate_qc_report(results_df, output_path):
    """Generate quality control report with visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Dice score distribution
    axes[0, 0].hist(results_df['Dice'], bins=20, color='steelblue', edgecolor='black')
    axes[0, 0].axvline(0.7, color='orange', linestyle='--', label='Good threshold')
    axes[0, 0].axvline(0.85, color='green', linestyle='--', label='Excellent threshold')
    axes[0, 0].set_xlabel('Dice Coefficient')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Dice Scores')
    axes[0, 0].legend()
    
    # 2. Dice vs IoU scatter
    axes[0, 1].scatter(results_df['Dice'], results_df['IoU'], alpha=0.6)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Perfect correlation')
    axes[0, 1].set_xlabel('Dice Coefficient')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('Dice vs IoU Correlation')
    axes[0, 1].legend()
    
    # 3. Precision-Recall scatter
    axes[1, 0].scatter(results_df['Recall'], results_df['Precision'], 
                       c=results_df['Dice'], cmap='viridis', alpha=0.6)
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision vs Recall (colored by Dice)')
    plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0], label='Dice')
    
    # 4. Quality categories
    categories = pd.cut(results_df['Dice'], 
                       bins=[0, 0.7, 0.85, 1.0],
                       labels=['Needs Review', 'Good', 'Excellent'])
    category_counts = categories.value_counts()
    
    axes[1, 1].bar(range(len(category_counts)), category_counts.values, 
                   color=['red', 'orange', 'green'])
    axes[1, 1].set_xticks(range(len(category_counts)))
    axes[1, 1].set_xticklabels(category_counts.index, rotation=45)
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Segmentation Quality Distribution')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Quality control report saved: {output_path}")
    
    # Print summary
    print("\n=== Quality Control Summary ===")
    print(f"Total cases: {len(results_df)}")
    print(f"\nQuality breakdown:")
    for cat, count in category_counts.items():
        pct = (count / len(results_df)) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

# Generate report if we have results
if len(df_results) > 0:
    generate_qc_report(df_results, '../dice/results/quality_control_report.png')
```

## Next Steps

1. **Annotate Real Data**: Use CVAT or Label Studio to create ground truth masks
2. **Train ML Model**: Use annotated data to train segmentation model
3. **Validate**: Use this toolkit to validate model predictions
4. **Iterate**: Refine annotations and model based on Dice scores

## Resources

- [CVAT Installation](https://opencv.github.io/cvat/docs/)
- [SAM Download](https://github.com/facebookresearch/segment-anything)
- [Medical Image Segmentation Best Practices](https://arxiv.org/abs/1904.03882)
