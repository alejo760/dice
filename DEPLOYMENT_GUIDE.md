# 🚀 Deployment Guide: Pneumonia Annotation Tool

This guide explains how to deploy the annotation app to **Hugging Face Spaces** (recommended, free) or **Streamlit Community Cloud**.

---

## 📋 Prerequisites

1. GitHub account
2. Hugging Face account (free) — [huggingface.co](https://huggingface.co)
3. Your annotation app files

---

## 📁 Step 1: Prepare Your Project Structure

Create a new folder with the following structure:

```
pneumonia-annotation-tool/
├── app.py                    # Your main Streamlit app (renamed)
├── requirements.txt          # Python dependencies
├── README.md                 # Project description
├── .gitignore                # Files to ignore
├── packages.txt              # System dependencies (optional)
└── sample_images/            # Sample X-ray images for demo (optional)
    └── sample_xray.jpg
```

### 1.1 Copy and rename your app

```bash
cp annotation_app.py app.py
```

### 1.2 Create `requirements.txt`

```txt
streamlit>=1.28.0
streamlit-drawable-canvas>=0.9.3
opencv-python-headless>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
```

> ⚠️ Use `opencv-python-headless` instead of `opencv-python` for cloud deployment!

### 1.3 Create `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*.so
.Python
env/
venv/
.venv/

# IDE
.vscode/
.idea/

# Data (don't upload patient data!)
data/Pacientes/
*.png
*.jpg
!sample_images/*.jpg

# OS
.DS_Store
Thumbs.db
```

### 1.4 Create `README.md`

```markdown
---
title: Pneumonia Consolidation Annotation Tool
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# 🫁 Pneumonia Consolidation Annotation Tool

A Streamlit application for radiologists to annotate pneumonia consolidations
on chest X-rays, supporting multilobar annotations with different colors.

## Features
- 🎨 Draw directly on X-ray images
- 🫁 Multiple consolidation sites with unique colors
- 🔍 Zoom & pan for detailed annotation
- 💾 Save masks and metadata
- 📊 Inter-rater agreement (Dice, IoU)
```

---

## 🤗 Option A: Deploy to Hugging Face Spaces (Recommended)

### Step 1: Create a Hugging Face Account
Go to [huggingface.co](https://huggingface.co) and sign up (free).

### Step 2: Create a New Space

1. Click your profile → **New Space**
2. Fill in:
   - **Space name**: `pneumonia-annotation-tool`
   - **License**: Choose one (e.g., MIT)
   - **SDK**: Select **Streamlit**
   - **Visibility**: Public or Private
3. Click **Create Space**

### Step 3: Upload Files

**Option A: Via Web Interface**
1. Go to your Space → **Files** tab
2. Click **Add file** → **Upload files**
3. Upload: `app.py`, `requirements.txt`, `README.md`

**Option B: Via Git (recommended)**

```bash
# Clone your Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/pneumonia-annotation-tool
cd pneumonia-annotation-tool

# Copy your files
cp /path/to/annotation_app.py app.py
cp /path/to/requirements.txt .
cp /path/to/README.md .

# Add and push
git add .
git commit -m "Initial deployment"
git push
```

### Step 4: Modify App for Cloud

Edit `app.py` to change the default patients path:

```python
# Change this line in the sidebar:
patients_path = st.sidebar.text_input(
    "Patients Folder Path",
    value="./uploaded_images",  # Changed from ../data/Pacientes
    help="Upload images or specify folder path",
)
```

### Step 5: Add Image Upload Feature

Add this code after the patients_path input to allow users to upload their own images:

```python
# Add file uploader for cloud deployment
st.sidebar.header("📤 Upload Images")
uploaded_files = st.sidebar.file_uploader(
    "Upload X-ray Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    upload_dir = Path("./uploaded_images/patient_upload")
    upload_dir.mkdir(parents=True, exist_ok=True)
    for uf in uploaded_files:
        with open(upload_dir / uf.name, "wb") as f:
            f.write(uf.getbuffer())
    st.sidebar.success(f"✅ Uploaded {len(uploaded_files)} images!")
```

### Step 6: Wait for Build

Hugging Face will automatically build and deploy. Check the **Logs** tab if there are errors.

🎉 **Your app will be live at**: `https://huggingface.co/spaces/YOUR_USERNAME/pneumonia-annotation-tool`

---

## ☁️ Option B: Deploy to Streamlit Community Cloud

### Step 1: Push to GitHub

```bash
# Create a new GitHub repository
# Then push your code:
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/pneumonia-annotation-tool.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Select:
   - Repository: `YOUR_USERNAME/pneumonia-annotation-tool`
   - Branch: `main`
   - Main file path: `app.py`
5. Click **Deploy**

🎉 **Your app will be live at**: `https://your-app-name.streamlit.app`

---

## 📤 Handling Patient Images

### For Privacy/HIPAA Compliance:

> ⚠️ **NEVER upload real patient data to public cloud services!**

### Options for Images:

| Method | Best For |
|--------|----------|
| **File Upload Widget** | Users upload their own images at runtime |
| **Private Space** | Hugging Face private Spaces (requires Pro) |
| **Self-hosted** | Run on your own server with patient data |
| **Sample Images** | Include anonymized/synthetic samples for demo |

### Adding Sample Images for Demo

1. Create `sample_images/` folder in your repo
2. Add anonymized or synthetic X-ray images
3. Update the default path:

```python
patients_path = st.sidebar.text_input(
    "Patients Folder Path",
    value="./sample_images",
)
```

---

## 🔧 Complete Modified `app.py` for Cloud

Here are the key modifications needed for cloud deployment:

### 1. Add imports at the top:
```python
import os
from pathlib import Path
```

### 2. Replace the patients path section with:
```python
# ── Sidebar: patients path ─────────────────────────────────────────
st.sidebar.header("📁 Patient Data")

# File uploader for cloud deployment
st.sidebar.subheader("📤 Upload X-rays")
uploaded_files = st.sidebar.file_uploader(
    "Upload chest X-ray images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload JPG/PNG chest X-ray images to annotate"
)

# Create upload directory
upload_dir = Path("./uploaded_images/uploads")
upload_dir.mkdir(parents=True, exist_ok=True)

if uploaded_files:
    for uf in uploaded_files:
        file_path = upload_dir / uf.name
        with open(file_path, "wb") as f:
            f.write(uf.getbuffer())
    st.sidebar.success(f"✅ {len(uploaded_files)} image(s) ready!")

patients_path = st.sidebar.text_input(
    "Images Folder Path",
    value="./uploaded_images",
    help="Folder containing images (or use uploader above)",
)
```

### 3. Update `get_all_patient_images` function:

```python
def get_all_patient_images(base_path):
    """Scan folders and collect all JPG/PNG images with annotation status."""
    base = Path(base_path)
    patient_images = []
    if not base.exists():
        return patient_images

    # Get all subdirectories (including 'uploads')
    folders = [base] + [f for f in base.iterdir() if f.is_dir()]
    
    for folder in folders:
        img_files = sorted(
            list(folder.glob("*.jpg")) + 
            list(folder.glob("*.JPG")) +
            list(folder.glob("*.jpeg")) +
            list(folder.glob("*.png"))
        )
        for img in img_files:
            if "_mask" in img.name:  # Skip mask files
                continue
            mask_path = img.parent / f"{img.stem}_mask.png"
            meta_path = img.parent / f"{img.stem}_annotation.json"
            patient_images.append({
                "patient_id": folder.name,
                "image_path": img,
                "image_name": img.name,
                "mask_path": mask_path,
                "metadata_path": meta_path,
                "annotated": mask_path.exists(),
            })
    return patient_images
```

---

## 📝 Quick Deployment Checklist

- [ ] Rename `annotation_app.py` → `app.py`
- [ ] Create `requirements.txt` with `opencv-python-headless`
- [ ] Create `README.md` with Hugging Face metadata
- [ ] Add file upload widget for cloud users
- [ ] Update default paths for cloud environment
- [ ] Remove/anonymize any patient data
- [ ] Test locally with `streamlit run app.py`
- [ ] Push to GitHub or Hugging Face
- [ ] Verify deployment and check logs

---

## 🆘 Troubleshooting

### "No module named cv2"
→ Use `opencv-python-headless` in requirements.txt

### "Permission denied" errors
→ Use relative paths (`./uploaded_images`) not absolute paths

### App crashes on large images
→ Add memory limit handling or resize images on upload

### Slow startup
→ Reduce dependencies, use `@st.cache_data` for heavy computations

---

## 📞 Support

- **Hugging Face Docs**: [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Streamlit Community**: [discuss.streamlit.io](https://discuss.streamlit.io)

---

*Created for the Pneumonia Consolidation Annotation Tool - Hospital Alma Máter*
