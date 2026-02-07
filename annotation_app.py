"""
Ground Truth Annotation Tool for Radiologists - Pneumonia Consolidation

Features:
1. Browse patient X-ray images from Pacientes folder automatically
2. Annotate consolidation directly in the browser (no external tools)
3. Multiple consolidation entries for multilobar pneumonia
4. Save mask + metadata JSON in the same patient folder
5. Progress tracking, inter-rater comparison, zoom, dark theme
"""

import sys
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
import json
import io
from datetime import datetime
from streamlit_drawable_canvas import st_canvas
import hashlib

# ============================================================================
# AUTHENTICATION
# ============================================================================

# User credentials (username: hashed_password)
# To add users: hash their password with hashlib.sha256("password".encode()).hexdigest()
USERS = {
    "admin": "6ef53576c614c35328ec86075d78cde376aa6b87504b39798d2ce4962a5a621a",
    "daniel": "dcabf0e5cfa74308f61ed0bcb7bd5565cfe6d890bb3e2ff7528d588da6f9c623",
}


def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def check_credentials(username: str, password: str) -> bool:
    """Verify username and password."""
    if username in USERS:
        return USERS[username] == hash_password(password)
    return False


def login_form():
    """Display login form and handle authentication."""
    st.set_page_config(
        page_title="Login - Annotation Tool",
        page_icon="🔐",
        layout="centered",
    )
    
    st.title("🔐 Login")
    st.markdown("### Pneumonia Consolidation Annotation Tool")
    st.markdown("---")
    
    with st.form("login_form"):
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
        submit = st.form_submit_button("🚀 Login", use_container_width=True)
        
        if submit:
            if check_credentials(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
    
    st.markdown("---")
    st.caption("Contact administrator for access credentials.")


def logout_button():
    """Display logout button in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"👤 Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()

# ============================================================================
# CONSOLIDATION COLOR PALETTE (one distinct color per site)
# ============================================================================

CONSOLIDATION_COLORS = [
    ("#00FF00", "Lime"),
    ("#FF4444", "Red"),
    ("#4488FF", "Blue"),
    ("#FFD700", "Gold"),
    ("#FF69B4", "Pink"),
    ("#00FFFF", "Cyan"),
    ("#FF8C00", "Orange"),
    ("#9370DB", "Purple"),
    ("#32CD32", "Green2"),
    ("#FF1493", "DeepPink"),
]


def get_color_for_index(idx: int) -> tuple:
    """Return (hex_color, label) for a given consolidation index."""
    return CONSOLIDATION_COLORS[idx % len(CONSOLIDATION_COLORS)]


# ============================================================================
# METRIC FUNCTIONS
# ============================================================================

def calculate_dice_coefficient(mask1, mask2):
    m1 = (mask1 > 0).astype(np.uint8)
    m2 = (mask2 > 0).astype(np.uint8)
    intersection = np.sum(m1 * m2)
    total = np.sum(m1) + np.sum(m2)
    if total == 0:
        return 1.0
    return (2.0 * intersection) / total


def calculate_iou(mask1, mask2):
    m1 = (mask1 > 0).astype(np.uint8)
    m2 = (mask2 > 0).astype(np.uint8)
    intersection = np.sum(m1 * m2)
    union = np.sum(m1) + np.sum(m2) - intersection
    if union == 0:
        return 1.0
    return intersection / union


def calculate_precision_recall(ground_truth, prediction):
    gt = (ground_truth > 0).astype(np.uint8)
    pred = (prediction > 0).astype(np.uint8)
    tp = np.sum(gt * pred)
    fp = np.sum((1 - gt) * pred)
    fn = np.sum(gt * (1 - pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


# ============================================================================
# IMAGE & DATA HELPERS
# ============================================================================

def load_image_from_path(image_path):
    """Load image as RGB numpy array (original, no CLAHE).
    
    Uses PIL as primary method for better cloud compatibility,
    with cv2 as fallback for edge cases.
    """
    image_path = Path(image_path)
    
    # Method 1: Use PIL (more reliable for cloud/uploaded files)
    try:
        with Image.open(image_path) as pil_img:
            # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            img = np.array(pil_img)
            return img
    except Exception as e:
        pass  # Fall through to cv2 method
    
    # Method 2: Fallback to OpenCV
    try:
        img = cv2.imread(str(image_path))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    
    # Method 3: Read bytes directly (for cloud file systems)
    try:
        with open(image_path, 'rb') as f:
            file_bytes = f.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    
    return None


def scale_image_preserve_ratio(img, target_width=900):
    """Scale image so width = target_width, preserving aspect ratio."""
    h, w = img.shape[:2]
    ratio = target_width / w
    new_h = int(h * ratio)
    new_w = target_width
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return scaled, ratio


def get_all_patient_images(base_path):
    """Scan patient folders and collect all JPG/PNG images with annotation status."""
    base = Path(base_path)
    patient_images = []
    if not base.exists():
        return patient_images

    # Get all subdirectories (including 'uploads' folder for cloud mode)
    folders = [base] + [f for f in base.iterdir() if f.is_dir()]
    
    for folder in folders:
        img_files = sorted(
            list(folder.glob("*.jpg")) + 
            list(folder.glob("*.JPG")) +
            list(folder.glob("*.jpeg")) +
            list(folder.glob("*.png"))
        )
        for img in img_files:
            # Skip mask files
            if "_mask" in img.name:
                continue
            mask_path = img.parent / f"{img.stem}_mask.png"
            meta_path = img.parent / f"{img.stem}_annotation.json"
            # Use folder name as patient_id, or 'uploaded' for root
            patient_id = folder.name if folder != base else "uploaded"
            patient_images.append({
                "patient_id": patient_id,
                "image_path": img,
                "image_name": img.name,
                "mask_path": mask_path,
                "metadata_path": meta_path,
                "annotated": mask_path.exists(),
            })
    return patient_images


def get_annotation_progress(patient_images):
    total = len(patient_images)
    done = sum(1 for img in patient_images if img["annotated"])
    pct = (done / total * 100) if total > 0 else 0
    return done, total, pct


def save_annotation_in_patient_folder(
    image_path, mask_array, annotator_name, metadata_dict, original_shape
):
    """Save mask (rescaled to original size) + metadata JSON in patient folder."""
    image_path = Path(image_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resize mask back to original image dimensions
    orig_h, orig_w = original_shape[:2]
    mask_resized = cv2.resize(
        mask_array, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    )

    mask_filename = f"{image_path.stem}_mask.png"
    mask_path = image_path.parent / mask_filename
    cv2.imwrite(str(mask_path), mask_resized)

    metadata = {
        "image_name": image_path.name,
        "patient_id": image_path.parent.name,
        "annotator": annotator_name,
        "timestamp": timestamp,
        "mask_file": mask_filename,
        **metadata_dict,
    }
    meta_path = image_path.parent / f"{image_path.stem}_annotation.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return mask_path, meta_path


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # ── Authentication Check ───────────────────────────────────────────
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        login_form()
        return
    
    # ── Authenticated: Show main app ───────────────────────────────────
    st.set_page_config(
        page_title="Ground Truth Annotation Tool",
        page_icon="🫁",
        layout="wide",
    )

    st.title("🫁 Pneumonia Consolidation — Ground Truth Annotation")
    
    # Show logout button
    logout_button()

    # ── Sidebar: annotator ─────────────────────────────────────────────
    st.sidebar.header("👤 Annotator")
    annotator_name = st.sidebar.text_input("Your Name / ID", value="Radiologist1")

    # ── Sidebar: patients path ─────────────────────────────────────────
    st.sidebar.header("📁 Patient Data")
    
    # Image upload for cloud deployment
    st.sidebar.subheader("📤 Upload X-rays")
    uploaded_files = st.sidebar.file_uploader(
        "Upload chest X-ray images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload JPG/PNG chest X-ray images to annotate",
    )
    
    # Create upload directory
    upload_dir = Path("./uploaded_images")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    if uploaded_files:
        for uf in uploaded_files:
            # Use filename (without extension) as patient ID
            file_path = upload_dir / uf.name
            with open(file_path, "wb") as f:
                f.write(uf.getbuffer())
        st.sidebar.success(f"✅ {len(uploaded_files)} image(s) uploaded!")
    
    st.sidebar.divider()
    
    patients_path = st.sidebar.text_input(
        "Images Folder Path",
        value="./uploaded_images",
        help="Folder with images (use uploader above for cloud, or local path)",
    )

    # ── Load images ────────────────────────────────────────────────────
    patient_images = get_all_patient_images(patients_path)

    if not patient_images:
        st.error(
            f"No JPG images found under **{patients_path}**. "
            "Check the path and ensure folders contain .jpg files."
        )
        return

    # ── Sidebar: progress ──────────────────────────────────────────────
    annotated_count, total_count, progress_pct = get_annotation_progress(
        patient_images
    )
    st.sidebar.header("📊 Progress")
    st.sidebar.progress(progress_pct / 100)
    st.sidebar.metric("Annotated", f"{annotated_count} / {total_count}")
    st.sidebar.metric("Completion", f"{progress_pct:.1f}%")

    # ── Sidebar: filter ────────────────────────────────────────────────
    st.sidebar.header("🔍 Filter")
    show_filter = st.sidebar.radio(
        "Show",
        ["All Images", "Not Annotated", "Annotated"],
        index=1,
    )
    if show_filter == "Not Annotated":
        filtered_images = [i for i in patient_images if not i["annotated"]]
    elif show_filter == "Annotated":
        filtered_images = [i for i in patient_images if i["annotated"]]
    else:
        filtered_images = patient_images

    if not filtered_images:
        st.warning(f"No images match filter **{show_filter}**.")
        return

    # ── Navigation state ───────────────────────────────────────────────
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if st.session_state.current_index >= len(filtered_images):
        st.session_state.current_index = 0
    current_image = filtered_images[st.session_state.current_index]

    # ── Sidebar: drawing settings ──────────────────────────────────────
    st.sidebar.header("🎨 Drawing Settings")
    stroke_width = st.sidebar.slider("Brush Size", 1, 50, 15)

    drawing_mode = st.sidebar.selectbox(
        "Drawing Tool",
        ["freedraw", "rect", "circle", "line"],
        index=0,
        help="freedraw: freehand brush · rect/circle/line: shapes",
    )

    canvas_width = st.sidebar.slider(
        "Canvas Width (px)", 600, 1400, 900, step=50,
        help="Adjust to fit your screen. Aspect ratio is always preserved.",
    )

    # ── Sidebar: zoom controls ─────────────────────────────────────────
    st.sidebar.header("🔍 Zoom & Pan")

    # Initialise zoom state
    if "zoom_level" not in st.session_state:
        st.session_state.zoom_level = 1.0
    if "zoom_pan_x" not in st.session_state:
        st.session_state.zoom_pan_x = 0.5
    if "zoom_pan_y" not in st.session_state:
        st.session_state.zoom_pan_y = 0.5

    # Quick zoom buttons
    zb1, zb2, zb3, zb4 = st.sidebar.columns(4)
    with zb1:
        if st.button("➖", key="zoom_out", help="Zoom out",
                     use_container_width=True):
            st.session_state.zoom_level = max(
                1.0, round(st.session_state.zoom_level - 0.25, 2)
            )
            st.rerun()
    with zb2:
        if st.button("➕", key="zoom_in", help="Zoom in",
                     use_container_width=True):
            st.session_state.zoom_level = min(
                5.0, round(st.session_state.zoom_level + 0.25, 2)
            )
            st.rerun()
    with zb3:
        if st.button("🔄", key="zoom_reset", help="Reset zoom",
                     use_container_width=True):
            st.session_state.zoom_level = 1.0
            st.session_state.zoom_pan_x = 0.5
            st.session_state.zoom_pan_y = 0.5
            st.rerun()
    with zb4:
        st.write(f"**{st.session_state.zoom_level:.1f}x**")

    zoom_level = st.sidebar.slider(
        "Zoom Level", 1.0, 5.0, st.session_state.zoom_level, step=0.25,
        help="Drag or use ➕/➖ buttons above.",
        key="zoom_slider",
    )
    if zoom_level != st.session_state.zoom_level:
        st.session_state.zoom_level = zoom_level

    if st.session_state.zoom_level > 1.0:
        # Pan controls — arrows + sliders
        st.sidebar.caption("**Pan (arrow buttons or sliders)**")
        pa1, pa2, pa3 = st.sidebar.columns([1, 1, 1])
        pan_step = 0.08
        with pa1:
            if st.button("⬅️", key="pan_left", use_container_width=True):
                st.session_state.zoom_pan_x = max(
                    0.0, round(st.session_state.zoom_pan_x - pan_step, 2)
                )
                st.rerun()
            if st.button("⬆️", key="pan_up", use_container_width=True):
                st.session_state.zoom_pan_y = max(
                    0.0, round(st.session_state.zoom_pan_y - pan_step, 2)
                )
                st.rerun()
        with pa2:
            if st.button("➡️", key="pan_right", use_container_width=True):
                st.session_state.zoom_pan_x = min(
                    1.0, round(st.session_state.zoom_pan_x + pan_step, 2)
                )
                st.rerun()
            if st.button("⬇️", key="pan_down", use_container_width=True):
                st.session_state.zoom_pan_y = min(
                    1.0, round(st.session_state.zoom_pan_y + pan_step, 2)
                )
                st.rerun()
        with pa3:
            st.write(
                f"x={st.session_state.zoom_pan_x:.2f}\n"
                f"y={st.session_state.zoom_pan_y:.2f}"
            )

        zoom_pan_x = st.sidebar.slider(
            "Pan H", 0.0, 1.0, st.session_state.zoom_pan_x, step=0.05,
            key="pan_h_slider",
        )
        zoom_pan_y = st.sidebar.slider(
            "Pan V", 0.0, 1.0, st.session_state.zoom_pan_y, step=0.05,
            key="pan_v_slider",
        )
        if zoom_pan_x != st.session_state.zoom_pan_x:
            st.session_state.zoom_pan_x = zoom_pan_x
        if zoom_pan_y != st.session_state.zoom_pan_y:
            st.session_state.zoom_pan_y = zoom_pan_y
    else:
        st.session_state.zoom_pan_x = 0.5
        st.session_state.zoom_pan_y = 0.5

    zoom_pan_x = st.session_state.zoom_pan_x
    zoom_pan_y = st.session_state.zoom_pan_y
    zoom_level = st.session_state.zoom_level

    # ── Tabs ───────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📋 Annotate", "🔄 Compare", "📖 Guidelines"])

    # ================================================================
    # TAB 1 — ANNOTATE
    # ================================================================
    with tab1:
        # Navigation bar
        nav1, nav2, nav3, nav4, nav5 = st.columns([1, 1, 3, 1, 1])

        with nav1:
            if st.button("⬅️ Previous", use_container_width=True):
                if st.session_state.current_index > 0:
                    st.session_state.current_index -= 1
                    st.rerun()
        with nav2:
            if st.button("Next ➡️", use_container_width=True):
                if st.session_state.current_index < len(filtered_images) - 1:
                    st.session_state.current_index += 1
                    st.rerun()
        with nav3:
            st.info(
                f"Image **{st.session_state.current_index + 1}** of "
                f"**{len(filtered_images)}** · Patient "
                f"**{current_image['patient_id']}**"
            )
        with nav4:
            jump_to = st.number_input(
                "Go to #",
                min_value=1,
                max_value=len(filtered_images),
                value=st.session_state.current_index + 1,
                key="jump",
            )
            if jump_to - 1 != st.session_state.current_index:
                st.session_state.current_index = jump_to - 1
                st.rerun()
        with nav5:
            if current_image["annotated"]:
                st.success("✅ Done")
            else:
                st.warning("⏳ Pending")

        st.divider()

        # Load original image (NO CLAHE)
        img_rgb = load_image_from_path(current_image["image_path"])
        if img_rgb is None:
            st.error(f"Cannot load image: {current_image['image_path']}")
            # Debug info for cloud troubleshooting
            with st.expander("🔧 Debug Info"):
                st.write(f"**Path:** `{current_image['image_path']}`")
                st.write(f"**Exists:** {Path(current_image['image_path']).exists()}")
                if Path(current_image['image_path']).exists():
                    st.write(f"**Size:** {Path(current_image['image_path']).stat().st_size} bytes")
                st.write(f"**Python:** {sys.version}")
                st.write(f"**OpenCV:** {cv2.__version__}")
                import PIL
                st.write(f"**Pillow:** {PIL.__version__}")
            return

        # Debug: Show image info (can be removed in production)
        # st.caption(f"Image loaded: {img_rgb.shape}, dtype={img_rgb.dtype}")

        # Scale image to canvas_width preserving aspect ratio
        img_scaled, scale_ratio = scale_image_preserve_ratio(img_rgb, canvas_width)

        # Apply zoom: crop a region of the scaled image and enlarge it
        if zoom_level > 1.0:
            zh, zw = img_scaled.shape[:2]
            crop_h = int(zh / zoom_level)
            crop_w = int(zw / zoom_level)
            # Calculate crop origin from pan sliders
            max_y = zh - crop_h
            max_x = zw - crop_w
            start_y = int(zoom_pan_y * max_y)
            start_x = int(zoom_pan_x * max_x)
            img_cropped = img_scaled[
                start_y : start_y + crop_h,
                start_x : start_x + crop_w,
            ]
            # Resize cropped region back to canvas dimensions
            img_for_canvas = cv2.resize(
                img_cropped, (zw, zh), interpolation=cv2.INTER_LINEAR
            )
        else:
            img_for_canvas = img_scaled
            start_x, start_y, crop_w, crop_h = (
                0, 0, img_scaled.shape[1], img_scaled.shape[0]
            )

        canvas_h, canvas_w = img_for_canvas.shape[:2]

        # Ensure image is in correct format for PIL/canvas (uint8 RGB)
        if img_for_canvas.dtype != np.uint8:
            img_for_canvas = img_for_canvas.astype(np.uint8)
        if len(img_for_canvas.shape) == 2:  # Grayscale
            img_for_canvas = cv2.cvtColor(img_for_canvas, cv2.COLOR_GRAY2RGB)
        elif img_for_canvas.shape[2] == 4:  # RGBA
            img_for_canvas = cv2.cvtColor(img_for_canvas, cv2.COLOR_RGBA2RGB)
        
        # Create PIL Image for canvas background
        pil_background = Image.fromarray(img_for_canvas, mode='RGB')

        st.subheader(
            f"Patient {current_image['patient_id']} — "
            f"{current_image['image_name']}"
        )

        col_canvas, col_meta = st.columns([3, 1])

        # ── Canvas ─────────────────────────────────────────────────────
        with col_canvas:
            # How many consolidation sites exist?
            state_key_preview = (
                f"consol_{current_image['patient_id']}_"
                f"{current_image['image_name']}"
            )
            n_sites = 1
            if state_key_preview in st.session_state:
                n_sites = max(1, len(st.session_state[state_key_preview]))

            # ── Site picker (controls stroke colour only) ──────────────
            # Always render the selectbox (even with 1 site) so
            # that the widget tree structure stays stable and the
            # canvas below never gets remounted / loses drawings.
            if "active_site" not in st.session_state:
                st.session_state.active_site = 0

            active_site = st.selectbox(
                "🫁 Active Consolidation Site (pick colour to draw)",
                list(range(n_sites)),
                format_func=lambda i: (
                    f"Site {i + 1} — {get_color_for_index(i)[1]}"
                ),
                index=min(
                    st.session_state.active_site, n_sites - 1
                ),
                key="site_picker",
            )
            st.session_state.active_site = active_site

            # Active site colour
            active_hex, active_label = get_color_for_index(active_site)
            r_c = int(active_hex[1:3], 16)
            g_c = int(active_hex[3:5], 16)
            b_c = int(active_hex[5:7], 16)
            fill_rgba = f"rgba({r_c}, {g_c}, {b_c}, 0.3)"

            # Build colour legend
            color_legend_parts = []
            for ci in range(n_sites):
                hex_c, label = get_color_for_index(ci)
                marker = "▶" if ci == active_site else "⬤"
                color_legend_parts.append(
                    f'<span style="color:{hex_c};font-weight:bold;">'
                    f'{marker} Site {ci + 1}</span>'
                )
            st.markdown(
                " &nbsp; ".join(color_legend_parts),
                unsafe_allow_html=True,
            )

            if zoom_level > 1.0:
                st.write(
                    f"**🎨 Drawing with {active_label} colour** "
                    f"(🔍 {zoom_level:.1f}x — Scroll ↕ to zoom, "
                    f"use arrow buttons to pan)"
                )
            else:
                st.write(
                    f"**🎨 Drawing with {active_label} colour** "
                    f"(Scroll ↕ over image to zoom)"
                )

            # ONE canvas per image — all sites draw here.
            # Only zoom/pan changes the key; switching active site
            # just changes the stroke colour, keeping all drawings.
            canvas_result = st_canvas(
                fill_color=fill_rgba,
                stroke_width=stroke_width,
                stroke_color=active_hex,
                background_image=pil_background,
                background_color="#000000",
                update_streamlit=True,
                height=canvas_h,
                width=canvas_w,
                drawing_mode=drawing_mode,
                key=f"canvas_{current_image['patient_id']}_"
                    f"{current_image['image_name']}_z{zoom_level}_"
                    f"x{zoom_pan_x}_y{zoom_pan_y}",
            )

            # --- Mouse-wheel zoom via JS injection ------------------
            import streamlit.components.v1 as components
            components.html(
                """
                <script>
                (function() {
                    // Find the Streamlit canvas elements
                    const doc = window.parent.document;
                    const canvases = doc.querySelectorAll(
                        'canvas[id*="canvas"]'
                    );
                    // Also listen on the overall app container
                    const appContainer = doc.querySelector(
                        '[data-testid="stAppViewContainer"]'
                    ) || doc.body;

                    function handleWheel(e) {
                        // Only act when scrolling over the canvas area
                        const target = e.target;
                        const isCanvas = (
                            target.tagName === 'CANVAS' ||
                            target.closest('.stCanvasContainer') ||
                            target.closest('[data-testid="stImage"]')
                        );
                        if (!isCanvas) return;

                        e.preventDefault();
                        e.stopPropagation();

                        // deltaY > 0 = scroll down = zoom out
                        const direction = e.deltaY > 0 ? 'out' : 'in';

                        // Find the zoom +/- buttons
                        const buttons = doc.querySelectorAll('button');
                        let targetBtn = null;
                        for (const btn of buttons) {
                            const txt = btn.textContent.trim();
                            if (direction === 'in' && txt === '➕') {
                                targetBtn = btn;
                                break;
                            }
                            if (direction === 'out' && txt === '➖') {
                                targetBtn = btn;
                                break;
                            }
                        }
                        if (targetBtn) {
                            targetBtn.click();
                        }
                    }

                    // Attach with capture to intercept before scroll
                    appContainer.addEventListener(
                        'wheel', handleWheel, {passive: false, capture: true}
                    );
                })();
                </script>
                """,
                height=0,
            )

            # Show thumbnail with zoom rectangle when zoomed in
            if zoom_level > 1.0:
                st.caption("📍 Overview — red box shows current zoom region")
                thumb_w = 250
                thumb, _ = scale_image_preserve_ratio(img_scaled, thumb_w)
                thumb_h_actual = thumb.shape[0]
                # Draw rectangle on thumbnail showing zoomed area
                th_ratio = thumb_w / img_scaled.shape[1]
                rx1 = int(start_x * th_ratio)
                ry1 = int(start_y * th_ratio)
                rx2 = int((start_x + crop_w) * th_ratio)
                ry2 = int((start_y + crop_h) * th_ratio)
                thumb_copy = thumb.copy()
                cv2.rectangle(thumb_copy, (rx1, ry1), (rx2, ry2),
                              (255, 0, 0), 2)
                st.image(thumb_copy, width=thumb_w)

        # ── Metadata column ───────────────────────────────────────────
        with col_meta:
            st.write("**📝 Annotation Metadata**")

            # Load existing metadata if any
            existing_metadata = {}
            if current_image["metadata_path"].exists():
                try:
                    with open(current_image["metadata_path"], "r") as f:
                        existing_metadata = json.load(f)
                except Exception:
                    pass

            # ── Multilobar consolidations ──────────────────────────────
            st.write("**🫁 Consolidation Sites**")

            location_options = [
                "Right Upper Lobe",
                "Right Middle Lobe",
                "Right Lower Lobe",
                "Left Upper Lobe",
                "Left Lower Lobe",
                "Lingula",
            ]
            type_options = [
                "Solid Consolidation",
                "Ground Glass Opacity",
                "Air Bronchograms",
                "Pleural Effusion",
                "Mixed",
            ]

            # Initialise session-state list for consolidations
            state_key = (
                f"consol_{current_image['patient_id']}_"
                f"{current_image['image_name']}"
            )
            if state_key not in st.session_state:
                # Pre-fill from existing metadata
                saved = existing_metadata.get("consolidations", [])
                if saved:
                    st.session_state[state_key] = saved
                else:
                    st.session_state[state_key] = [
                        {"location": "Right Lower Lobe",
                         "type": "Solid Consolidation"}
                    ]

            consolidations = st.session_state[state_key]

            # Render each consolidation entry
            for idx, entry in enumerate(consolidations):
                site_hex, site_label = get_color_for_index(idx)
                with st.expander(
                    f"⬤ Site {idx + 1}: {entry['location']}  "
                    f"({site_label})",
                    expanded=True,
                ):
                    loc = st.selectbox(
                        "Location",
                        location_options,
                        index=(
                            location_options.index(entry["location"])
                            if entry["location"] in location_options
                            else 0
                        ),
                        key=f"loc_{state_key}_{idx}",
                    )
                    ctype = st.selectbox(
                        "Type",
                        type_options,
                        index=(
                            type_options.index(entry["type"])
                            if entry["type"] in type_options
                            else 0
                        ),
                        key=f"type_{state_key}_{idx}",
                    )
                    consolidations[idx] = {"location": loc, "type": ctype}

                    if len(consolidations) > 1:
                        if st.button(
                            "🗑️ Remove", key=f"rm_{state_key}_{idx}",
                            use_container_width=True,
                        ):
                            consolidations.pop(idx)
                            st.rerun()

            if st.button("➕ Add Another Consolidation Site",
                         use_container_width=True):
                consolidations.append(
                    {"location": "Left Lower Lobe",
                     "type": "Solid Consolidation"}
                )
                # Auto-switch to the new site so the next strokes
                # use the new colour immediately
                st.session_state.active_site = len(consolidations) - 1
                st.rerun()

            st.divider()

            # Pattern summary
            involved_lobes = list({c["location"] for c in consolidations})
            if len(involved_lobes) >= 2:
                st.info(
                    f"🔴 **Multilobar** pneumonia — "
                    f"{len(involved_lobes)} lobes involved"
                )
            else:
                st.info(f"🟡 **Unilobar** — {involved_lobes[0]}")

            confidence = st.slider(
                "Confidence",
                min_value=1,
                max_value=5,
                value=existing_metadata.get("confidence", 5),
            )
            notes = st.text_area(
                "Clinical Notes",
                value=existing_metadata.get("clinical_notes", ""),
                placeholder="E.g., Silhouette sign present, bilateral involvement",
            )

            # Drawn area stats
            if canvas_result.image_data is not None:
                alpha = canvas_result.image_data[:, :, 3]
                drawn_px = int(np.sum(alpha > 0))
                total_px = alpha.shape[0] * alpha.shape[1]
                if drawn_px > 0:
                    st.metric(
                        "Drawn Area",
                        f"{(drawn_px / total_px) * 100:.2f}%",
                    )
                    st.metric("Pixels", f"{drawn_px:,}")

            st.divider()

            # ── Save / Delete ──────────────────────────────────────────
            b1, b2 = st.columns(2)

            def _build_metadata():
                return {
                    "consolidations": consolidations,
                    "involved_lobes": involved_lobes,
                    "multilobar": len(involved_lobes) >= 2,
                    "confidence": confidence,
                    "clinical_notes": notes,
                }

            with b1:
                if st.button(
                    "💾 Save & Next", type="primary",
                    use_container_width=True,
                ):
                    if (
                        canvas_result.image_data is not None
                        and np.sum(canvas_result.image_data[:, :, 3] > 0) > 0
                    ):
                        mask = canvas_result.image_data[:, :, 3]
                        save_annotation_in_patient_folder(
                            current_image["image_path"],
                            mask,
                            annotator_name,
                            _build_metadata(),
                            img_rgb.shape,
                        )
                        st.success("✅ Saved!")
                        if (
                            st.session_state.current_index
                            < len(filtered_images) - 1
                        ):
                            st.session_state.current_index += 1
                        st.rerun()
                    else:
                        st.error("Please draw an annotation first!")

            with b2:
                if st.button("💾 Save Only", use_container_width=True):
                    if (
                        canvas_result.image_data is not None
                        and np.sum(canvas_result.image_data[:, :, 3] > 0) > 0
                    ):
                        mask = canvas_result.image_data[:, :, 3]
                        save_annotation_in_patient_folder(
                            current_image["image_path"],
                            mask,
                            annotator_name,
                            _build_metadata(),
                            img_rgb.shape,
                        )
                        st.success("✅ Saved!")
                    else:
                        st.error("Please draw an annotation first!")

            if current_image["annotated"]:
                if st.button("🗑️ Delete Annotation",
                             use_container_width=True):
                    if current_image["mask_path"].exists():
                        current_image["mask_path"].unlink()
                    if current_image["metadata_path"].exists():
                        current_image["metadata_path"].unlink()
                    st.success("Annotation deleted!")
                    st.rerun()

            # ── Download Buttons ───────────────────────────────────────
            st.divider()
            st.write("**📥 Download Annotation**")
            
            # Generate file ID from patient_id and image name
            file_id = f"{current_image['patient_id']}_{current_image['image_path'].stem}"
            
            # Download mask
            if (
                canvas_result.image_data is not None
                and np.sum(canvas_result.image_data[:, :, 3] > 0) > 0
            ):
                # Create mask from current canvas
                mask_data = canvas_result.image_data[:, :, 3]
                # Resize to original image dimensions
                orig_h, orig_w = img_rgb.shape[:2]
                mask_resized = cv2.resize(
                    mask_data, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )
                
                # Encode mask as PNG
                _, mask_buffer = cv2.imencode(".png", mask_resized)
                mask_bytes = mask_buffer.tobytes()
                
                # Create JSON metadata
                metadata_download = {
                    "image_id": file_id,
                    "image_name": current_image["image_name"],
                    "patient_id": current_image["patient_id"],
                    "annotator": annotator_name,
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "consolidations": consolidations,
                    "involved_lobes": involved_lobes,
                    "multilobar": len(involved_lobes) >= 2,
                    "confidence": confidence,
                    "clinical_notes": notes,
                    "mask_dimensions": {"width": orig_w, "height": orig_h},
                    "annotated_pixels": int(np.sum(mask_resized > 0)),
                    "annotated_area_percent": float(
                        np.sum(mask_resized > 0) / (orig_w * orig_h) * 100
                    ),
                }
                json_bytes = json.dumps(metadata_download, indent=2).encode("utf-8")
                
                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        label="📥 Download Mask (PNG)",
                        data=mask_bytes,
                        file_name=f"{file_id}_mask.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                with dl2:
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_bytes,
                        file_name=f"{file_id}_annotation.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                
                st.caption(f"Files will be named: `{file_id}_mask.png` and `{file_id}_annotation.json`")
            
            elif current_image["annotated"] and current_image["mask_path"].exists():
                # Load existing saved annotation for download
                existing_mask = cv2.imread(
                    str(current_image["mask_path"]), cv2.IMREAD_GRAYSCALE
                )
                if existing_mask is not None:
                    _, mask_buffer = cv2.imencode(".png", existing_mask)
                    mask_bytes = mask_buffer.tobytes()
                    
                    # Load existing JSON
                    if current_image["metadata_path"].exists():
                        with open(current_image["metadata_path"], "r") as f:
                            existing_json = json.load(f)
                        json_bytes = json.dumps(existing_json, indent=2).encode("utf-8")
                    else:
                        json_bytes = b"{}"
                    
                    dl1, dl2 = st.columns(2)
                    with dl1:
                        st.download_button(
                            label="📥 Download Saved Mask",
                            data=mask_bytes,
                            file_name=f"{file_id}_mask.png",
                            mime="image/png",
                            use_container_width=True,
                        )
                    with dl2:
                        st.download_button(
                            label="📥 Download Saved JSON",
                            data=json_bytes,
                            file_name=f"{file_id}_annotation.json",
                            mime="application/json",
                            use_container_width=True,
                        )
                    st.caption(f"Files: `{file_id}_mask.png` / `{file_id}_annotation.json`")
            else:
                st.info("Draw an annotation to enable downloads")

    # ================================================================
    # TAB 2 — COMPARE
    # ================================================================
    with tab2:
        st.header("Compare Annotations Between Radiologists")

        cmp1, cmp2 = st.columns(2)
        with cmp1:
            st.subheader("Radiologist 1")
            mask1_file = st.file_uploader(
                "Upload Mask 1", type=["png"], key="comp1"
            )
            name1 = st.text_input("Name", value="Radiologist 1", key="name1")
        with cmp2:
            st.subheader("Radiologist 2")
            mask2_file = st.file_uploader(
                "Upload Mask 2", type=["png"], key="comp2"
            )
            name2 = st.text_input("Name", value="Radiologist 2", key="name2")

        if mask1_file and mask2_file:
            mask1 = cv2.imdecode(
                np.frombuffer(mask1_file.read(), np.uint8),
                cv2.IMREAD_GRAYSCALE,
            )
            mask1_file.seek(0)
            mask2 = cv2.imdecode(
                np.frombuffer(mask2_file.read(), np.uint8),
                cv2.IMREAD_GRAYSCALE,
            )
            mask2_file.seek(0)

            if mask1.shape != mask2.shape:
                mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]))

            dice = calculate_dice_coefficient(mask1, mask2)
            iou = calculate_iou(mask1, mask2)
            precision, recall = calculate_precision_recall(mask1, mask2)

            st.subheader("📊 Inter-Rater Agreement")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Dice", f"{dice:.4f}")
            m2.metric("IoU", f"{iou:.4f}")
            m3.metric("Precision", f"{precision:.4f}")
            m4.metric("Recall", f"{recall:.4f}")

            if dice >= 0.80:
                st.success("✅ Excellent Agreement")
            elif dice >= 0.70:
                st.info("ℹ️ Good Agreement")
            elif dice >= 0.50:
                st.warning("⚠️ Fair Agreement — Review recommended")
            else:
                st.error("❌ Poor Agreement — Consensus needed")

            st.subheader("Visual Comparison")
            overlay = np.zeros(
                (mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8
            )
            overlay[mask1 > 0] = [0, 255, 0]
            overlap = (mask1 > 0) & (mask2 > 0)
            overlay[mask2 > 0] = [255, 0, 0]
            overlay[overlap] = [255, 255, 0]
            st.image(
                overlay,
                caption=(
                    f"Green: {name1} | Red: {name2} | Yellow: Agreement"
                ),
                use_column_width=True,
            )

    # ================================================================
    # TAB 3 — GUIDELINES
    # ================================================================
    with tab3:
        st.header("📖 Annotation Guidelines")
        st.markdown(
            """
### What to Annotate

**Pneumonia consolidation** appears as white / opaque areas where air
spaces are filled with fluid.

### Multilobar Pneumonia

When consolidation is present in **more than one lobe**, add a separate
consolidation entry for each affected site using the **"➕ Add Another
Consolidation Site"** button. This lets us track multilobar involvement
accurately.

### Key Radiologic Signs

#### ✅ Include in Your Mask
1. **Air Bronchograms** — Dark branching tubes inside consolidation
2. **Silhouette Sign** — Heart / diaphragm border lost in consolidation
3. **Solid Consolidation** — Dense white opaque areas
4. **Ground Glass Opacity** — Subtle hazy areas at edges

#### ❌ Exclude from Your Mask
1. **Ribs** — Trace "through" rib shadows
2. **Normal lung tissue** — Don't over-segment
3. **Pleural effusion** (unless asked) — Smooth meniscus sign

### Drawing Tools
| Tool | Use for |
|---|---|
| **freedraw** | Freehand tracing of consolidation borders |
| **rect** | Quick rectangular ROI |
| **circle** | Circular / oval regions |
| **line** | Straight edge tracing |

### Colors
Each consolidation site is automatically assigned a **unique colour**
(Lime, Red, Blue, Gold, …). Select the active site before drawing
so annotations are visually distinguishable.

### Tips
1. **Draw directly** on the canvas — no external tools needed
2. **Adjust brush size** with the sidebar slider
3. **Zoom**: scroll ↕ your mouse wheel over the image, or use the
   ➕ / ➖ buttons in the sidebar
4. **Pan**: when zoomed in, use the arrow buttons (⬅️➡️⬆️⬇️) or
   sliders to navigate
5. **Be consistent** — same criteria for every image

### Quality Metrics
| Dice Score | Interpretation |
|---|---|
| > 0.80 | ✅ Excellent agreement |
| 0.70 – 0.80 | Good agreement |
| < 0.70 | ⚠️ Needs review / consensus |
"""
        )


if __name__ == "__main__":
    main()
