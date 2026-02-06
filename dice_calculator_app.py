"""
Streamlit App for Pneumonia Consolidation Ground Truth Annotation

This app allows radiologists to:
1. Load and view chest X-ray images with enhancement
2. Create segmentation masks using drawing tools
3. Save ground truth annotations
4. Compare annotations between radiologists (inter-rater agreement)
5. Export annotations for ML training
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
import io
import zipfile
import json
from datetime import datetime
from streamlit_drawable_canvas import st_canvas


def calculate_dice_coefficient(mask1, mask2):
    """
    Calculate Dice coefficient between two binary masks.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        mask1: First binary mask (numpy array)
        mask2: Second binary mask (numpy array)
        
    Returns:
        Dice coefficient (float between 0 and 1)
    """
    # Ensure masks are binary
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.sum(mask1 * mask2)
    mask1_sum = np.sum(mask1)
    mask2_sum = np.sum(mask2)
    
    # Handle edge case where both masks are empty
    if mask1_sum + mask2_sum == 0:
        return 1.0
    
    dice = (2.0 * intersection) / (mask1_sum + mask2_sum)
    return dice


def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU / Jaccard Index).
    
    IoU = |A ∩ B| / |A ∪ B|
    """
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2) - intersection
    
    if union == 0:
        return 1.0
    
    return intersection / union


def calculate_precision_recall(ground_truth, prediction):
    """
    Calculate precision and recall for segmentation.
    
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    """
    gt = (ground_truth > 0).astype(np.uint8)
    pred = (prediction > 0).astype(np.uint8)
    
    tp = np.sum(gt * pred)
    fp = np.sum((1 - gt) * pred)
    fn = np.sum(gt * (1 - pred))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall


def calculate_hausdorff_distance(mask1, mask2):
    """
    Calculate Hausdorff distance between two masks.
    Measures the maximum distance from a point in one set to the closest point in the other set.
    """
    from scipy.spatial.distance import directed_hausdorff
    
    # Get coordinates of mask pixels
    coords1 = np.argwhere(mask1 > 0)
    coords2 = np.argwhere(mask2 > 0)
    
    if len(coords1) == 0 or len(coords2) == 0:
        return float('inf')
    
    # Calculate directed Hausdorff distances
    d1 = directed_hausdorff(coords1, coords2)[0]
    d2 = directed_hausdorff(coords2, coords1)[0]
    
    # Return maximum
    return max(d1, d2)


def create_overlay_visualization(image, ground_truth, prediction, alpha=0.5):
    """
    Create visualization with ground truth (green) and prediction (red) overlaid.
    Overlap areas appear in yellow.
    """
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Normalize image to 0-255 range
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    # Create colored masks
    overlay = image.copy()
    
    # Ground truth in green
    gt_mask = (ground_truth > 0)
    overlay[gt_mask] = overlay[gt_mask] * (1 - alpha) + np.array([0, 255, 0]) * alpha
    
    # Prediction in red
    pred_mask = (prediction > 0)
    overlay[pred_mask] = overlay[pred_mask] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    
    # Overlap in yellow (green + red)
    overlap = gt_mask & pred_mask
    overlay[overlap] = overlay[overlap] * (1 - alpha) + np.array([255, 255, 0]) * alpha
    
    return overlay.astype(np.uint8)


def create_comparison_grid(image, ground_truth, prediction, overlay):
    """
    Create a 2x2 grid showing all visualizations.
    """
    # Ensure all images are the same size and RGB
    h, w = image.shape[:2]
    
    # Convert masks to RGB
    gt_rgb = cv2.cvtColor((ground_truth * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    pred_rgb = cv2.cvtColor((prediction * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image
    
    # Create grid
    top_row = np.hstack([image_rgb, gt_rgb])
    bottom_row = np.hstack([pred_rgb, overlay])
    grid = np.vstack([top_row, bottom_row])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    
    cv2.putText(grid, 'Original', (10, 30), font, font_scale, color, thickness)
    cv2.putText(grid, 'Ground Truth', (w + 10, 30), font, font_scale, color, thickness)
    cv2.putText(grid, 'Prediction', (10, h + 30), font, font_scale, color, thickness)
    cv2.putText(grid, 'Overlay', (w + 10, h + 30), font, font_scale, color, thickness)
    
    return grid


def load_image(uploaded_file):
    """Load and convert uploaded image to numpy array."""
    image = Image.open(uploaded_file)
    load_and_enhance_image(uploaded_file, enhance=True):
    """Load image and optionally apply enhancement for better visualization."""
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    if enhance:
        # Apply CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_gray)
        
        # Sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    return img_gray


def save_annotation(image_name, mask, annotator_name, notes=""):
    """Save annotation with metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create annotations directory
    annotations_dir = Path("annotations/ground_truth")
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Save mask
    mask_filename = f"{Path(image_name).stem}_{annotator_name}_{timestamp}.png"
    mask_path = annotations_dir / mask_filename
    cv2.imwrite(str(mask_path), mask)
    
    # Save metadata
    metadata = {
        "image_name": image_name,
        "annotator": annotator_name,
        "timestamp": timestamp,
        "notes": notes,
        "mask_file": mask_filename
    }
    
    metadata_path = annotations_dir / f"{Path(image_name).stem}_{annotator_name}_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return mask_path, metadata_path


def load_existing_annotations(image_name):
    """Load all existing annotations for an image."""
    annotations_dir = Path("annotations/ground_truth")
    if not annotations_dir.exists():
        return []
    
    image_stem = Path(image_name).stem
    annotations = []
    
    for mask_file in annotations_dir.glob(f"{image_stem}_*.png"):
        # Find corresponding metadata
        json_file = mask_file.with_suffix('.json')
        if json_file.exists():
            with open(json_file, 'r') as f:
                metadata = json.load(f)
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                annotations.append({
                    'mask': mask,
                    'metadata': metadata,
                    'file': mask_file
                })
    
    return annotations


def main():
    st.set_page_config(
        page_title="Ground Truth Annotation Tool - Pneumonia",
        page_icon="🫁",
        layout="wide"
    )
    
    st.title("🫁 Pneumonia Consolidation Ground Truth Annotation Tool")
    st.markdown("""
    ### For Radiologists - Create Expert Annotations
    
    This tool helps you create high-quality ground truth segmentation masks for pneumonia consolidation.
    
    **Workflow:**
    1. 📁 Load a chest X-ray image
    2. 🎨 Draw consolidation masks using the annotation tools
    3. 💾 Save your annotation with notes
    4. 🔄 Compare with other radiologists' annotations (optional
    - 🟢 Green: Ground Truth
    - 🔴 Red: Prediction
    - 🟡 Yellow: Overlap (correct prediction)
    """)
    
    # Sidebar for settings
    st.sidebar.header("👤 Annotator Information")
    annotator_name = st.sidebar.text_input("Your Name/ID", value="Radiologist1", 
                                           help="Enter your name or ID for tracking")
    
    st.sidebar.header("🎨 Annotation Settings")
    stroke_width = st.sidebar.slider("Brush Size", 1, 50, 10)
    stroke_color = st.sidebar.color_picker("Drawing Color", "#FFFF00")
    
    st.sidebar.header("🖼️ Display Settings")
    enhast.header("📋 Create New Annotation")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Chest X-ray Image (JPG/PNG)", 
            type=['jpg', 'jpeg', 'png'],
            key='xray_upload'
        )
        
        with col3:
            st.subheader("Predicted Mask")
            prediction_file = st.file_uploader(
                "Upload Prediction Mask", 
                type=['jpg', 'jpeg', 'png'],
                key='prediction'
            )
        
        if original_file and ground_truth_file and prediction_file:
            # Load and process image
            img_array = load_and_enhance_image(uploaded_file, enhance=enhance_image)
            
            # Display image info
            st.info(f"📸 Image: {uploaded_file.name} | Size: {img_array.shape[1]}x{img_array.shape[0]} pixels")
            
            # Check for existing annotations
            existing_annotations = load_existing_annotations(uploaded_file.name)
            if existing_annotations:
                st.warning(f"⚠️ Found {len(existing_annotations)} existing annotation(s) for this image")
                if st.checkbox("Load previous annotation to edit"):
                    selected_annotation = st.selectbox(
                        "Select annotation to load",
                        range(len(existing_annotations)),
                        format_func=lambda i: f"{existing_annotations[i]['metadata']['annotator']} - {existing_annotations[i]['metadata']['timestamp']}"
                    )
                    # Load the selected mask as initial drawing
                    initial_mask = existing_annotations[selected_annotation]['mask']
                else:
                    initial_mask = None
            else:
                initial_mask = None
            
            # Annotation interface
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🎨 Draw Consolidation Mask")
                
                if show_guidelines:
                    st.info("""
                    **Quick Tips:**
                    - ✅ Include air bronchograms (dark tubes in white area)
                    - ✅ Trace where heart/diaphragm border disappears
                    - ❌ Don't include ribs in mask
                    - Use eraser to refine fuzzy borders
                    """)
                
                # Convert image to RGB for canvas
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                
                # Drawing canvas
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 0, 0.3)",
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_image=Image.fromarray(img_rgb),
                    update_streamlit=True,
                    height=img_array.shape[0],
                    width=img_array.shape[1],
                    drawing_mode="freedraw",
                    initial_drawing=initial_mask,
                    key="canvas",
                )
            
            with col2:
                st.subheader("📝 Annotation Details")
                
                # Consolidation characteristics
                st.write("**Consolidation Type:**")
                consolidation_type = st.multiselect(
                    "Select all that apply",
                    ["Solid Consolidation", "Ground Glass Opacity", "Air Bronchograms", "Pleural Effusion"],
                    default=["Solid Consolidation"]
                )
                
                # Location
                location = st.selectbox(
                    "Location",
                    ["Right Upper Lobe", "Right Middle Lobe", "Right Lower Lobe",
                     "Left Upper Lobe", "Left Lower Lobe", "Bilateral", "Other"]
                )
                
                # Confidence
                confidence = st.slider("Annotation Confidence", 1, 5, 5,
                                     help="How confident are you in this segmentation?")
                
                # Notes
                notes = st.text_area(
                    "Clinical Notes (optional)",
                    placeholder="E.g., 'Silhouette sign present with right heart border'"
                )
                
                # Statistics
                if canvas_result.image_data is not None:
                    mask = canvas_result.image_data[:, :, 3] > 0
                    mask_area = np.sum(mask)
                    total_area = mask.shape[0] * mask.shape[1]
                    percentage = (mask_area / total_area) * 100
                    
                    st.metric("Annotated Area", f"{mask_area} px²")
                    st.metric("Coverage", f"{percentage:.2f}%")
            
            # Save annotation
            st.divider()
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("💾 Save Annotation", type="primary", use_container_width=True):
                    if canvas_result.image_data is not None:
                        # Extract mask from canvas
                        mask = canvas_result.image_data[:, :, 3]
                        
                        # Create metadata
                        full_notes = {
                            "consolidation_type": consolidation_type,
                            "location": location,
                            "confidence": confidence,
                            "clinical_notes": notes
                        }
                        
                        # Save
                        mask_path, metadata_path = save_annotation(
                            uploaded_file.name,
                            mask,
                            annotator_name,
                            json.dumps(full_notes)
                        )
                        
                        st.success(f"✅ Annotation saved!\n- Mask: {mask_path.name}\n- Metadata: {metadata_path.name}")
                    else:
                        st.error("❌ No annotation drawn yet!")
            
            with col2:
                if st.button("🗑️ Clear Canvas", use_container_width=True):
                    st.rerun()
            
            with col3:
                if canvas_result.image_data is not None:
                    # Download current mask
                    mask = canvas_result.image_data[:, :, 3]
                    mask_pil = Image.fromarray(mask)
                    buf = io.BytesIO()
                    mask_pil.save(buf, format='PNG')
                    
                    st.download_button(
                        label="⬇️ Download Current Mask",
                        data=buf.getvalue(),
                        file_name=f"{Path(uploaded_file.name).stem}_mask_{annotator_name}.png",
                        mime="image/png",
                        use_container_width=True
                    )
    
    with tab2:
        st.header("🔄 Compare Annotations Between Radiologists")
        st.markdown("Calculate inter-rater agreement (Dice coefficient) between two radiologists' annotations of the same image.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Radiologist 1 Annotation")
            mask1_file = st.file_uploader("Upload Mask 1", type=['png', 'jpg'], key='compare_mask1')
            annotator1 = st.text_input("Annotator 1 Name", value="Radiologist 1")
        
        with col2:
            st.subheader("Radiologist 2 Annotation")
            mask2_file = st.file_uploader("Upload Mask 2", type=['png', 'jpg'], key='compare_mask2')
            annotator2 = st.text_input("Annotator 2 Name", value="Radiologist 2")
        
        if mask1_file and mask2_file:
            mask1 = cv2.imread(io.BytesIO(mask1_file.read()), cv2.IMREAD_GRAYSCALE) if isinstance(mask1_file, st.runtime.uploaded_file_manager.UploadedFile) else cv2.imdecode(np.frombuffer(mask1_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            mask2 = cv2.imread(io.BytesIO(mask2_file.read()), cv2.IMREAD_GRAYSCALE) if isinstance(mask2_file, st.runtime.uploaded_file_manager.UploadedFile) else cv2.imdecode(np.frombuffer(mask2_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            
            mask1_file.seek(0)
            mask2_file.seek(0)
            mask1 = cv2.imdecode(np.frombuffer(mask1_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            mask1_file.seek(0)
            mask2_file.seek(0)
            mask2 = cv2.imdecode(np.frombuffer(mask2_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            mask2_file.seek(0)
            
            # Resize if needed
            if mask1.shape != mask2.shape:
                mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]))
            
            # Calculate agreement metrics
            dice = calculate_dice_coefficient(mask1, mask2)
            iou = calculate_iou(mask1, mask2)
            precision, recall = calculate_precision_recall(mask1, mask2)
            
            # Display metrics
            st.subheader("📊 Inter-Rater Agreement")
            
            cols = st.columns(4)
            with cols[0]:
                st.metric("Dice Coefficient", f"{dice:.4f}")
            with cols[1]:
                st.metric("IoU", f"{iou:.4f}")
            with cols[2]:
                st.metric("Precision", f"{precision:.4f}")
            with cols[3]:
                st.metric("Recall", f"{recall:.4f}")
            
            # Interpretation
            if dice >= 0.80:
                st.success("✅ **Excellent Agreement** - Annotations are highly consistent")
            elif dice >= 0.70:
                st.info("ℹ️ **Good Agreement** - Acceptable consistency, may need minor discussion")
            elif dice >= 0.50:
                st.warning("⚠️ **Fair Agreement** - Significant differences, recommend consensus review")
            else:
                st.error("❌ **Poor Agreement** - Major differences, requires discussion and re-annotation")
            
            # Visual comparison
            st.subheader("🖼️ Visual Comparison")
            
            # Create overlay: Annotator1=green, Annotator2=red, Overlap=yellow
            overlay = np.zeros((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8)
            overlay[mask1 > 0] = [0, 255, 0]  # Green for annotator 1
            overlap = (mask1 > 0) & (mask2 > 0)
            overlay[mask2 > 0] = [255, 0, 0]  # Red for annotator 2
            overlay[overlap] = [255, 255, 0]  # Yellow for agreement
            
            st.image(overlay, caption=f"Green: {annotator1} only | Red: {annotator2} only | Yellow: Agreement", use_container_width=True)
            
            # Area statistics
            st.subheader("📐 Area Statistics")
            area1 = np.sum(mask1 > 0)
            area2 = np.sum(mask2 > 0)
            overlap_area = np.sum(overlap)
            
            data = {
                'Annotator': [annotator1, annotator2, 'Overlap'],
                'Area (pixels)': [area1, area2, overlap_area],
                'Percentage': [100, (area2/area1)*100, (overlap_area/area1)*100]
            }
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
    
    with tab3:
        st.header("📚 Browse Existing Annotations")
        
        annotations_dir = Path("annotations/ground_truth")
        if annotations_dir.exists():
            all_annotations = list(annotations_dir.glob("*.json"))
            
            if all_annotations:
                st.info(f"Found {len(all_annotations)} annotation(s)")
                
                # Group by image
                annotations_by_image = {}
                for json_file in all_annotations:
                    with open(json_file, 'r') as f:
                        metadata = json.load(f)
                        image_name = metadata['image_name']
                        if image_name not in annotations_by_image:
                            annotations_by_image[image_name] = []
                        annotations_by_image[image_name].append(metadata)
                
                # Display selector
                selected_image = st.selectbox("Select Image", list(annotations_by_image.keys()))
                
                if selected_image:
                    st.subheader(f"Annotations for: {selected_image}")
                    
                    annotations = annotations_by_image[selected_image]
                    
                    # Display as table
                    df_data = []
                    for ann in annotations:
                        df_data.append({
                            'Annotator': ann['annotator'],
                            'Timestamp': ann['timestamp'],
                            'Mask File': ann['mask_file']
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Show masks
                    cols = st.columns(min(len(annotations), 3))
                    for idx, ann in enumerate(annotations):
                        mask_path = annotations_dir / ann['mask_file']
                        if mask_path.exists():
                            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                            with cols[idx % 3]:
                                st.image(mask, caption=f"{ann['annotator']}", use_container_width=True)
                    
                    # Export options
                    st.divider()
                    st.subheader("📦 Export Annotations")
                    
                    if st.button("Export All Annotations as ZIP"):
                        # Create ZIP file
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for ann in annotations:
                                mask_file = annotations_dir / ann['mask_file']
                                json_file = annotations_dir / f"{Path(ann['mask_file']).stem}.json"
                                
                                if mask_file.exists():
                                    zip_file.write(mask_file, mask_file.name)
                                if json_file.exists():
                                    zip_file.write(json_file, json_file.name)
                        
                        st.download_button(
                            label="⬇️ Download ZIP",
                            data=zip_buffer.getvalue(),
                            file_name=f"annotations_{selected_image.replace('.', '_')}.zip",
                            mime="application/zip"
                        )
            else:
                st.warning("No annotations found. Create your first annotation in the 'Annotate' tab!")
        else:
            st.warning("Annotations directory not found. Create your first annotation to get started!")
    
    with tab4:
        st.markdown("""
        ## 📋 Annotation Guidelines for Pneumonia Consolidation
        
        ### What is Pneumonia Consolidation?
        Pneumonia consolidation appears as white/opaque areas in the lung fields on chest X-rays. 
        It represents areas where the air spaces are filled with fluid, pus, or cellular debris.
        
        ### Key Radiologic Signs to Look For:
        
        #### 1. **Air Bronchograms** ✓
        - Dark, branching tubes visible inside white consolidation
        - **100% diagnostic for consolidation**
        - Include entire region surrounding these patterns in your mask
        
        #### 2. **Silhouette Sign**
        - Heart or diaphragm border "disappears" into white area
        - Indicates consolidation is touching that structure
        - Include this boundary in your segmentation
        
        #### 3. **Border Characteristics**
        - Consolidations have "fuzzy" or poorly defined borders
        - Often blend into surrounding lung tissue
        - Use the enhanced preprocessing images to see borders better
        
        ### Annotation Best Practices:
        
        1. **Use Polygon Tool First**
           - Trace rough outline of consolidation
           - Don't worry about perfect edges initially
        
        2. **Refine with Brush Tool**
           - Clean up edges where consolidation fades
           - Use eraser for areas that are too included
        
        3. **Avoid Common Mistakes**
           - ❌ Don't include ribs in the mask
           - ❌ Don't over-segment into normal lung
           - ❌ Don't miss subtle ground-glass opacities at edges
           - ✓ Do trace "through" ribs mentally
           - ✓ Do include the full air bronchogram region
        
        4. **Different Types to Label**
           - **Solid Consolidation**: Dense white areas
           - **Ground Glass Opacity**: Subtle hazy areas
           - **Air Bronchograms**: The pattern itself confirms consolidation
        
        ### Quality Metrics Interpretation:
        
        - **Dice > 0.85**: Excellent segmentation
        - **Dice 0.70-0.85**: Good segmentation (acceptable for fuzzy borders)
        - **Dice < 0.70**: Needs review (may have missed area or over-segmented)
        
        ### Tips for Difficult Cases:
        
        1. **Behind Ribs**: Mentally interpolate the consolidation boundary through rib shadows
        2. **Near Heart**: Use silhouette sign - if heart border disappears, include that area
        3. **Multiple Patches**: Each separate consolidation should be in the same mask
        4. **Pleural Effusion vs Consolidation**: Effusion is smooth with meniscus; consolidation is irregular
        """)
        
        st.info("""
        **Recommended Workflow:**
        1. Preprocess image with enhancement script
        2. Annotate in CVAT or similar tool
        3. Use this app to validate against expert annotations
        4. Iterate until Dice > 0.80
        """)


if __name__ == "__main__":
    main()
