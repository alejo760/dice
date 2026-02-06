# Quick Start Guide - Get Started in 5 Minutes! ⚡

## For Immediate Testing (No Installation)

If you just want to see what the Dice calculator does:

### Option 1: Test with Sample Data
```bash
cd dice/

# Create a simple test mask
python3 -c "
import cv2
import numpy as np

# Create sample image and masks
img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
mask1 = np.zeros((512, 512), dtype=np.uint8)
mask2 = np.zeros((512, 512), dtype=np.uint8)

# Draw circles as "consolidations"
cv2.circle(mask1, (250, 250), 80, 255, -1)
cv2.circle(mask2, (260, 260), 75, 255, -1)  # Slightly different

# Save
cv2.imwrite('test_image.jpg', img)
cv2.imwrite('test_gt.png', mask1)
cv2.imwrite('test_pred.png', mask2)

print('Test files created!')
"
```

### Option 2: Use Your Own Data
If you already have:
- Chest X-ray image (JPG/PNG)
- Ground truth mask (JPG/PNG)
- Predicted mask (JPG/PNG)

Skip to the "Launch App" section below.

---

## Full Setup (Recommended)

### 1. Install Dependencies (2 minutes)
```bash
cd dice/
pip install -r requirements.txt
```

### 2. Launch the Dice Calculator App (30 seconds)
```bash
streamlit run dice_calculator_app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Upload Your Images
1. **Original Image**: Upload your chest X-ray
2. **Ground Truth Mask**: Upload the expert annotation
3. **Predicted Mask**: Upload the model/algorithm output

You'll instantly see:
- ✅ Dice coefficient
- ✅ IoU score
- ✅ Precision & Recall
- ✅ Visual overlays (green=GT, red=prediction, yellow=overlap)

---

## Common Use Cases

### Use Case 1: "I want to enhance my X-rays first"
```bash
# Single image
python preprocessing_consolidation.py \
    --input ../data/Pacientes/7035909/7035909_20240326.jpg \
    --output enhanced_xray.jpg

# All images in a folder
python preprocessing_consolidation.py \
    --input ../data/Pacientes/ \
    --output ./enhanced_images/ \
    --batch
```

### Use Case 2: "I want to compare two annotators"
```bash
# Launch app
streamlit run dice_calculator_app.py

# In the app:
# 1. Upload X-ray as "Original Image"
# 2. Upload Annotator 1's mask as "Ground Truth"
# 3. Upload Annotator 2's mask as "Prediction"
# 
# Dice > 0.80 = Good agreement
```

### Use Case 3: "I want automatic segmentation with SAM"
```bash
# First, download SAM checkpoint:
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Interactive mode (click on consolidation)
python sam_integration.py \
    --checkpoint sam_vit_h_4b8939.pth \
    --image chest_xray.jpg \
    --mode interactive

# Automatic batch processing
python sam_integration.py \
    --checkpoint sam_vit_h_4b8939.pth \
    --input_dir ../data/Pacientes/ \
    --output_dir ./sam_masks/ \
    --mode auto
```

### Use Case 4: "I want to calculate Dice for many images"
```bash
# Use the Streamlit app's "Batch Processing" tab
# Upload two ZIP files:
#   1. images.zip (contains all X-rays)
#   2. masks.zip (contains corresponding masks)
# 
# The app will process all pairs and generate CSV report
```

---

## Interpreting Results

### Dice Coefficient
| Score | Interpretation |
|-------|---------------|
| > 0.85 | **Excellent** - Ready for publication |
| 0.70 - 0.85 | **Good** - Acceptable for fuzzy boundaries |
| 0.50 - 0.70 | **Fair** - Needs review, may need re-annotation |
| < 0.50 | **Poor** - Significant disagreement |

### Visual Overlay Colors
- 🟢 **Green**: Ground truth only (missed by prediction)
- 🔴 **Red**: Prediction only (false positive)
- 🟡 **Yellow**: Overlap (correct prediction) ✓

**Goal**: Maximize yellow, minimize green and red

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install streamlit
```

### "App won't start"
```bash
# Check Python version (need 3.8+)
python --version

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Try with full path
python -m streamlit run dice_calculator_app.py
```

### "Images are different sizes"
Don't worry! The app automatically resizes masks to match the image.

### "SAM is too slow"
- Use smaller model: `--model_type vit_b` instead of `vit_h`
- Use GPU if available (automatically detected)
- Process images in smaller batches

---

## Next Steps After Quick Start

1. ✅ **You've tested the app** → Read [example_usage.md](example_usage.md)
2. ✅ **You've calculated some Dice scores** → Check [TODO.md](TODO.md) for project roadmap
3. ✅ **You're ready to annotate** → See annotation guidelines in [README.md](README.md)
4. ✅ **You want to train ML** → Use annotations as training data

---

## One-Line Commands for Copy-Paste

```bash
# Install everything
cd dice && pip install -r requirements.txt

# Launch app
streamlit run dice_calculator_app.py

# Enhance single image
python preprocessing_consolidation.py --input xray.jpg --output enhanced.jpg

# Batch enhance
python preprocessing_consolidation.py --input ../data/Pacientes/ --output ./enhanced/ --batch

# Test with sample data
python3 -c "import cv2; import numpy as np; img=np.random.randint(0,255,(512,512),dtype=np.uint8); m1=np.zeros((512,512),dtype=np.uint8); m2=np.zeros((512,512),dtype=np.uint8); cv2.circle(m1,(250,250),80,255,-1); cv2.circle(m2,(260,260),75,255,-1); cv2.imwrite('test_image.jpg',img); cv2.imwrite('test_gt.png',m1); cv2.imwrite('test_pred.png',m2); print('Test files created!')"
```

---

## Support

- 📖 **Full Documentation**: [README.md](README.md)
- 📝 **Examples**: [example_usage.md](example_usage.md)
- ✅ **Project Plan**: [TODO.md](TODO.md)
- 💬 **Questions**: Open an issue or contact project lead

**Ready to start? Run this now:**
```bash
streamlit run dice_calculator_app.py
```
