# Pneumonia Consolidation Segmentation Tools 🫁

A comprehensive toolkit for segmenting pneumonia consolidation in chest X-rays using machine learning, with tools for preprocessing, annotation, and validation using Dice coefficient metrics.

## 📋 Features

### 1. **Dice Score Calculator (Streamlit App)**
- Interactive web interface for calculating segmentation metrics
- Compare ground truth vs predicted masks
- Metrics: Dice coefficient, IoU, Precision, Recall, F1, Hausdorff distance
- Visual overlays with color-coded comparisons
- Batch processing support
- Built-in annotation guidelines

### 2. **Image Preprocessing**
- CLAHE enhancement for local contrast
- Sharpening filters to reveal air bronchograms
- Edge enhancement for consolidation boundaries
- Batch processing capabilities

### 3. **SAM Integration**
- Automatic mask generation using Segment Anything Model
- Interactive point-based segmentation
- Bounding box prompts
- Batch processing support

## 🚀 Quick Start

### Installation

```bash
# Clone or download this repository
cd dice/

# Install dependencies
pip install -r requirements.txt

# Optional: For SAM integration
# pip install segment-anything torch torchvision
# Download SAM checkpoint from: https://github.com/facebookresearch/segment-anything
```

### Running the Dice Calculator App

```bash
streamlit run dice_calculator_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Preprocessing Images

Enhance chest X-rays to better visualize consolidations:

```bash
# Single image
python preprocessing_consolidation.py \
    --input /path/to/image.jpg \
    --output /path/to/enhanced.jpg

# Batch processing
python preprocessing_consolidation.py \
    --input /path/to/images/ \
    --output /path/to/enhanced/ \
    --batch \
    --extension .jpg
```

### 2. Calculating Dice Scores

#### Using the Streamlit App:
1. Start the app: `streamlit run dice_calculator_app.py`
2. Upload your chest X-ray, ground truth mask, and predicted mask
3. View metrics and visualizations instantly
4. Download results as CSV or images

#### Programmatic Usage:

```python
import cv2
from dice_calculator_app import calculate_dice_coefficient

# Load masks
ground_truth = cv2.imread('ground_truth_mask.png', cv2.IMREAD_GRAYSCALE)
prediction = cv2.imread('predicted_mask.png', cv2.IMREAD_GRAYSCALE)

# Calculate Dice
dice = calculate_dice_coefficient(ground_truth, prediction)
print(f"Dice Coefficient: {dice:.4f}")
```

### 3. Using SAM for Automatic Segmentation

First, download a SAM checkpoint:
- [ViT-H (Huge)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) - Most accurate
- [ViT-L (Large)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) - Balanced
- [ViT-B (Base)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) - Fastest

```bash
# Interactive mode (click points to guide segmentation)
python sam_integration.py \
    --checkpoint sam_vit_h_4b8939.pth \
    --image chest_xray.jpg \
    --mode interactive

# Automatic batch processing
python sam_integration.py \
    --checkpoint sam_vit_h_4b8939.pth \
    --input_dir /path/to/images/ \
    --output_dir /path/to/masks/ \
    --mode auto
```

## 📊 Understanding the Metrics

### Dice Coefficient (Main Metric)
- **Range**: 0 (no overlap) to 1 (perfect overlap)
- **Formula**: `2 × |A ∩ B| / (|A| + |B|)`
- **Interpretation**:
  - **> 0.85**: Excellent segmentation
  - **0.70-0.85**: Good (acceptable for fuzzy consolidation borders)
  - **< 0.70**: Needs review

### IoU (Jaccard Index)
- **Range**: 0 to 1
- **Formula**: `|A ∩ B| / |A ∪ B|`
- More strict than Dice coefficient

### Precision & Recall
- **Precision**: How many predicted pixels are correct
- **Recall**: How many actual consolidation pixels were found

### Hausdorff Distance
- Measures maximum distance between mask boundaries
- Lower is better (masks are closer)

## 🎯 Annotation Guidelines

### Key Radiologic Signs

#### 1. **Air Bronchograms** ✓
- Dark, branching tubes inside white consolidation
- **100% diagnostic** for pneumonia
- Include entire surrounding region in mask

#### 2. **Silhouette Sign**
- Heart or diaphragm border "disappears" into white area
- Include boundary in segmentation

#### 3. **Border Characteristics**
- Fuzzy, poorly defined edges
- Blend into surrounding tissue
- Use enhanced preprocessing to see better

### Best Practices

✅ **DO:**
- Trace through ribs mentally
- Include full air bronchogram regions
- Use preprocessing to see subtle borders
- Label different types: solid, ground-glass, air bronchograms

❌ **DON'T:**
- Include ribs in masks
- Over-segment into normal lung
- Miss subtle ground-glass opacities

## 📁 Project Structure

```
dice/
├── dice_calculator_app.py          # Main Streamlit application
├── preprocessing_consolidation.py  # Image enhancement tools
├── sam_integration.py              # SAM integration
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── annotations/                    # Store annotation masks (create)
│   ├── ground_truth/
│   └── predictions/
├── enhanced_images/                # Preprocessed images (create)
└── results/                        # Dice scores and reports (create)
```

## 🔧 Advanced Configuration

### Streamlit App Settings

In the sidebar:
- **Overlay Transparency**: Adjust visualization opacity
- **Calculate Hausdorff Distance**: Enable for boundary distance metrics (slower)

### Preprocessing Parameters

Edit `preprocessing_consolidation.py` to adjust:
```python
clahe = cv2.createCLAHE(
    clipLimit=3.0,      # Increase for more contrast
    tileGridSize=(8,8)  # Smaller = more local enhancement
)
```

### SAM Parameters

In `sam_integration.py`, adjust:
```python
# Confidence threshold for automatic detection
if scores[0] > 0.8:  # Lower = more permissive

# Grid density for automatic sampling
grid_size = 5  # Increase for finer sampling
```

## 🔬 Workflow Recommendations

### For Manual Annotation:
1. **Preprocess** images with `preprocessing_consolidation.py`
2. **Annotate** in CVAT or Label Studio
3. **Validate** with Dice calculator app
4. **Iterate** until Dice > 0.80

### For ML Training:
1. **Generate initial masks** with SAM
2. **Refine manually** in annotation tool
3. **Calculate metrics** to ensure quality
4. **Use as training data** for your model

### For Validation Study:
1. Have multiple annotators segment images
2. Compare annotations using Dice calculator
3. Calculate inter-rater agreement
4. Establish ground truth consensus

## 📚 References

### Tools & Models
- [CVAT](https://github.com/opencv/cvat) - Computer Vision Annotation Tool
- [SAM](https://github.com/facebookresearch/segment-anything) - Segment Anything Model
- [Streamlit](https://streamlit.io/) - Web app framework

### Medical Context
- Air Bronchograms: Air-filled bronchi visible against consolidated lung
- Silhouette Sign: Loss of normal boundaries due to adjacent opacity
- Consolidation: Filling of air spaces with fluid/exudate in pneumonia

## 🐛 Troubleshooting

### App won't start
```bash
# Check Streamlit installation
pip install --upgrade streamlit

# Run with verbose logging
streamlit run dice_calculator_app.py --logger.level=debug
```

### SAM errors
```bash
# Ensure PyTorch is installed
pip install torch torchvision

# Download correct checkpoint for model type
# vit_h, vit_l, or vit_b must match checkpoint
```

### Image size issues
Images are automatically resized to match. For best results:
- Use same resolution for all images in a study
- Minimum 512x512 recommended
- Maximum 2048x2048 for performance

## 📝 License

This project is for research and educational purposes related to pneumonia diagnosis and medical image segmentation.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional metrics (Surface Dice, Boundary IoU)
- 3D visualization support
- Integration with DICOM files
- Multi-class segmentation support

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Note**: This tool is for research purposes. Always validate with clinical experts and follow appropriate medical imaging guidelines.
