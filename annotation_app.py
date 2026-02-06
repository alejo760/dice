"""
Herramienta de Anotación Ground Truth para Radiólogos - Consolidación por Neumonía

Características:
1. Subir imágenes de rayos X para anotación
2. Anotar consolidación directamente en el navegador (sin herramientas externas)
3. Múltiples entradas de consolidación para neumonía multilobar
4. Guardar máscara + JSON de metadatos y descargar como ZIP
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
import json
import io
import zipfile
import logging
import os
import shutil
from datetime import datetime
from streamlit_drawable_canvas import st_canvas
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        page_title="Iniciar Sesión - Herramienta de Anotación",
        page_icon="🔐",
        layout="centered",
    )
    
    st.title("🔐 Iniciar Sesión")
    st.markdown("### Herramienta de Anotación de Consolidación por Neumonía")
    st.markdown("---")
    
    with st.form("login_form"):
        username = st.text_input("👤 Usuario", placeholder="Ingrese su usuario")
        password = st.text_input("🔑 Contraseña", type="password", placeholder="Ingrese su contraseña")
        submit = st.form_submit_button("🚀 Ingresar", use_container_width=True)
        
        if submit:
            if check_credentials(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("❌ Usuario o contraseña inválidos")
    
    st.markdown("---")
    st.caption("Contacte al administrador para obtener credenciales de acceso.")


def logout_button():
    """Display logout button in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"👤 Conectado como: **{st.session_state.username}**")
    if st.sidebar.button("🚪 Cerrar Sesión", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()

# ============================================================================
# CONSOLIDATION COLOR PALETTE (one distinct color per site)
# ============================================================================

CONSOLIDATION_COLORS = [
    ("#00FF00", "Verde Lima"),
    ("#FF4444", "Rojo"),
    ("#4488FF", "Azul"),
    ("#FFD700", "Dorado"),
    ("#FF69B4", "Rosa"),
    ("#00FFFF", "Cian"),
    ("#FF8C00", "Naranja"),
    ("#9370DB", "Púrpura"),
    ("#32CD32", "Verde2"),
    ("#FF1493", "Fucsia"),
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
    """Load image as RGB numpy array (original, no CLAHE)."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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
    base = Path(base_path).resolve()  # Use absolute path for reliable comparison
    patient_images = []
    
    logger.info(f"get_all_patient_images called with: {base_path}")
    logger.info(f"Resolved base path: {base}")
    
    if not base.exists():
        logger.warning(f"Base path does not exist: {base}")
        return patient_images

    # Get all subdirectories (including 'uploads' folder for cloud mode)
    try:
        subdirs = [f for f in base.iterdir() if f.is_dir()]
        logger.info(f"Found {len(subdirs)} subdirectories: {[d.name for d in subdirs]}")
    except Exception as e:
        logger.error(f"Error listing subdirectories: {e}")
        subdirs = []
    
    folders = [base] + subdirs
    
    for folder in folders:
        logger.info(f"Scanning folder: {folder}")
        # Collect all image extensions (case-insensitive approach)
        img_files = []
        for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
            found = list(folder.glob(ext))
            logger.info(f"  Pattern {ext}: found {len(found)} files")
            img_files.extend(found)
        
        img_files = sorted(set(img_files))  # Remove duplicates and sort
        logger.info(f"  Total images in {folder.name}: {len(img_files)}")
        
        for img in img_files:
            # Skip mask files
            if "_mask" in img.name:
                logger.info(f"  Skipping mask file: {img.name}")
                continue
            mask_path = img.parent / f"{img.stem}_mask.png"
            meta_path = img.parent / f"{img.stem}_annotation.json"
            # Use folder name as patient_id, or 'uploaded' for root
            is_base_folder = folder.resolve() == base
            patient_id = "uploaded" if is_base_folder else folder.name
            logger.info(f"  Adding image: {img.name}, patient_id: {patient_id}")
            patient_images.append({
                "patient_id": patient_id,
                "image_path": img,
                "image_name": img.name,
                "mask_path": mask_path,
                "metadata_path": meta_path,
                "annotated": mask_path.exists(),
            })
    
    logger.info(f"Total patient images found: {len(patient_images)}")
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

    # ── Clear uploaded images and cache on fresh session start ─────────
    if "session_initialized" not in st.session_state:
        st.session_state.session_initialized = True
        
        # Clear Streamlit cache
        st.cache_data.clear()
        st.cache_resource.clear()
        logger.info("Cleared Streamlit cache")
        
        # Clear uploaded images folder
        upload_folder = Path("./uploaded_images")
        if upload_folder.exists():
            try:
                import shutil
                for item in upload_folder.iterdir():
                    if item.is_file():
                        item.unlink()
                        logger.info(f"Deleted file: {item.name}")
                    elif item.is_dir():
                        shutil.rmtree(item)
                        logger.info(f"Deleted folder: {item.name}")
                logger.info("Cleared all uploaded images on session start")
            except Exception as e:
                logger.error(f"Error clearing uploaded images: {e}")

    st.title("🫁 Pneumonia Consolidation — Ground Truth Annotation")
    
    # Show logout button
    logout_button()
    
    # Use logged-in username as annotator name
    annotator_name = st.session_state.username

    # ── Sidebar: Upload Images ─────────────────────────────────────────
    st.sidebar.header("📤 Subir Radiografías")
    uploaded_files = st.sidebar.file_uploader(
        "Subir imágenes de rayos X de tórax",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Suba imágenes JPG/PNG de rayos X para anotar",
    )
    
    # Create upload directory with absolute path for reliability
    upload_dir = Path("./uploaded_images").resolve()
    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Upload directory: {upload_dir}")
        logger.info(f"Upload directory exists: {upload_dir.exists()}")
        logger.info(f"Upload directory is writable: {os.access(upload_dir, os.W_OK)}")
    except Exception as e:
        logger.error(f"Failed to create upload directory: {e}")
        st.error(f"Error creating upload directory: {e}")
    
    # Track uploaded files to avoid infinite rerun loop
    if uploaded_files:
        logger.info(f"Received {len(uploaded_files)} files from uploader")
        new_files_uploaded = False
        for uf in uploaded_files:
            file_path = upload_dir / uf.name
            logger.info(f"Processing file: {uf.name} -> {file_path}")
            # Only write if file doesn't exist yet
            if not file_path.exists():
                try:
                    file_content = uf.getbuffer()
                    logger.info(f"File buffer size: {len(file_content)} bytes")
                    with open(file_path, "wb") as f:
                        f.write(file_content)
                    logger.info(f"Saved new file: {file_path}")
                    # Verify file was written
                    if file_path.exists():
                        logger.info(f"Verified file exists: {file_path}, size: {file_path.stat().st_size}")
                        new_files_uploaded = True
                    else:
                        logger.error(f"File was not saved: {file_path}")
                except Exception as e:
                    logger.error(f"Error saving file {uf.name}: {e}")
                    st.sidebar.error(f"Error saving {uf.name}: {e}")
            else:
                logger.info(f"File already exists: {file_path}")
        
        if new_files_uploaded:
            st.sidebar.success(f"✅ ¡{len(uploaded_files)} imagen(es) subida(s)!")
            logger.info("Triggering rerun after new file upload")
            st.rerun()  # Refresh to load the new images
    
    st.sidebar.divider()

    # ── Load images ────────────────────────────────────────────────────
    # Use same absolute path as upload_dir for consistency
    patients_path = str(upload_dir)
    
    # Debug: List contents of upload directory
    logger.info(f"Scanning for images in: {patients_path}")
    try:
        if Path(patients_path).exists():
            all_files = list(Path(patients_path).iterdir())
            logger.info(f"Files in upload directory ({len(all_files)} total): {[f.name for f in all_files]}")
            # Show file types
            for f in all_files:
                logger.info(f"  File: {f.name}, is_file: {f.is_file()}, suffix: {f.suffix}")
        else:
            logger.warning(f"Upload directory does not exist: {patients_path}")
    except Exception as e:
        logger.error(f"Error listing upload directory: {e}")
    
    patient_images = get_all_patient_images(patients_path)
    logger.info(f"Found {len(patient_images)} patient images")
    for img in patient_images:
        logger.info(f"  - {img['image_name']} (annotated: {img['annotated']})")

    if not patient_images:
        st.info(
            "📤 **Suba imágenes de rayos X** usando el panel lateral para comenzar a anotar."
        )
        st.markdown("---")
        st.markdown(
            """
            ### 📋 Instrucciones:
            1. **Suba sus imágenes** usando el cargador en el panel lateral izquierdo
            2. **Dibuje las consolidaciones** directamente sobre la imagen
            3. **Guarde la anotación** y descargue los archivos de salida (máscara PNG + JSON)
            """
        )
        return

    # ── Navigation state ───────────────────────────────────────────────
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if st.session_state.current_index >= len(patient_images):
        st.session_state.current_index = 0
    current_image = patient_images[st.session_state.current_index]

    # ── Sidebar: drawing settings ──────────────────────────────────────
    st.sidebar.header("🎨 Configuración de Dibujo")
    stroke_width = st.sidebar.slider("Tamaño del Pincel", 1, 50, 15)

    drawing_mode = st.sidebar.selectbox(
        "Herramienta de Dibujo",
        ["freedraw", "rect", "circle", "line"],
        index=0,
        help="freedraw: pincel libre · rect/circle/line: formas",
    )

    # Fixed canvas width
    canvas_width = 900

    # ── Image info ────────────────────────────────────────────────────
    col_info1, col_info2 = st.columns([3, 1])
    with col_info1:
        st.info(f"**Imagen:** {current_image['image_name']}")
    with col_info2:
        if current_image["annotated"]:
            st.success("✅ Anotada")
        else:
            st.warning("⏳ Pendiente")
    
    st.divider()

    # Load original image (NO CLAHE)
    img_rgb = load_image_from_path(current_image["image_path"])
    if img_rgb is None:
        st.error(f"No se puede cargar la imagen: {current_image['image_path']}")
        return

    # Scale image to canvas_width preserving aspect ratio
    img_scaled, scale_ratio = scale_image_preserve_ratio(img_rgb, canvas_width)
    img_for_canvas = img_scaled
    
    logger.info(f"Image loaded: shape={img_rgb.shape}, dtype={img_rgb.dtype}")
    logger.info(f"Image scaled: shape={img_for_canvas.shape}, dtype={img_for_canvas.dtype}")

    canvas_h, canvas_w = img_for_canvas.shape[:2]
    logger.info(f"Canvas dimensions: {canvas_w}x{canvas_h}")
    
    # Convert to PIL Image for canvas
    pil_image = Image.fromarray(img_for_canvas)
    logger.info(f"PIL Image: mode={pil_image.mode}, size={pil_image.size}")

    st.subheader(
        f"Paciente {current_image['patient_id']} — "
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
            "🫁 Sitio de Consolidación Activo (elija color para dibujar)",
            list(range(n_sites)),
            format_func=lambda i: (
                f"Sitio {i + 1} — {get_color_for_index(i)[1]}"
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
                f'{marker} Sitio {ci + 1}</span>'
            )
        st.markdown(
            " &nbsp; ".join(color_legend_parts),
            unsafe_allow_html=True,
        )

        st.write(f"**🎨 Dibujando con color {active_label}**")

        # ONE canvas per image — all sites draw here.
        # Switching active site just changes the stroke colour, keeping all drawings.
        canvas_result = st_canvas(
            fill_color=fill_rgba,
            stroke_width=stroke_width,
            stroke_color=active_hex,
            background_image=pil_image,
            background_color="#000000",
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode=drawing_mode,
            key=f"canvas_{current_image['patient_id']}_{current_image['image_name']}",
        )

    # ── Metadata column ───────────────────────────────────────────
    with col_meta:
        st.write("**📝 Metadatos de Anotación**")

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
                    {"location": "Lóbulo Inferior Derecho",
                     "type": "Consolidación Sólida"}
                ]

        consolidations = st.session_state[state_key]

        # Render each consolidation entry
        for idx, entry in enumerate(consolidations):
            site_hex, site_label = get_color_for_index(idx)
            with st.expander(
                f"⬤ Sitio {idx + 1}: {entry['location']}  "
                f"({site_label})",
                expanded=True,
            ):
                loc = st.selectbox(
                    "Ubicación",
                    location_options,
                    index=(
                        location_options.index(entry["location"])
                        if entry["location"] in location_options
                        else 0
                    ),
                    key=f"loc_{state_key}_{idx}",
                )
                ctype = st.selectbox(
                    "Tipo",
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
                        "🗑️ Eliminar", key=f"rm_{state_key}_{idx}",
                        use_container_width=True,
                    ):
                        consolidations.pop(idx)
                        st.rerun()

        if st.button("➕ Agregar Otro Sitio de Consolidación",
                     use_container_width=True):
            consolidations.append(
                {"location": "Lóbulo Inferior Izquierdo",
                 "type": "Consolidación Sólida"}
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
                f"🔴 Neumonía **Multilobar** — "
                f"{len(involved_lobes)} lóbulos afectados"
            )
        else:
            st.info(f"🟡 **Unilobar** — {involved_lobes[0]}")

        confidence = st.slider(
            "Confianza",
            min_value=1,
            max_value=5,
            value=existing_metadata.get("confidence", 5),
        )
        notes = st.text_area(
            "Notas Clínicas",
            value=existing_metadata.get("clinical_notes", ""),
            placeholder="Ej., Signo de silueta presente, afectación bilateral",
        )

        # Drawn area stats
        if canvas_result.image_data is not None:
            alpha = canvas_result.image_data[:, :, 3]
            drawn_px = int(np.sum(alpha > 0))
            total_px = alpha.shape[0] * alpha.shape[1]
            if drawn_px > 0:
                st.metric(
                    "Área Dibujada",
                    f"{(drawn_px / total_px) * 100:.2f}%",
                )
                st.metric("Píxeles", f"{drawn_px:,}")

        st.divider()

        # ── Save Button ──────────────────────────────────────────
        def _build_metadata():
            return {
                "consolidations": consolidations,
                "involved_lobes": involved_lobes,
                "multilobar": len(involved_lobes) >= 2,
                "confidence": confidence,
                "clinical_notes": notes,
            }

        if st.button("💾 Guardar Anotación", type="primary",
                     use_container_width=True):
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
                st.success("✅ ¡Guardado!")
                st.rerun()
            else:
                st.error("¡Por favor dibuje una anotación primero!")

        if current_image["annotated"]:
            if st.button("🗑️ Eliminar Anotación",
                         use_container_width=True):
                if current_image["mask_path"].exists():
                    current_image["mask_path"].unlink()
                if current_image["metadata_path"].exists():
                    current_image["metadata_path"].unlink()
                st.success("¡Anotación eliminada!")
                st.rerun()

        # ── Download All Button ───────────────────────────────────
        st.divider()
        st.write("**📥 Descargar Anotación**")
        
        # Generate file ID from patient_id and image name
        file_id = f"{current_image['patient_id']}_{current_image['image_path'].stem}"
        
        # Check if we have annotation data to download
        has_current_annotation = (
            canvas_result.image_data is not None
            and np.sum(canvas_result.image_data[:, :, 3] > 0) > 0
        )
        has_saved_annotation = (
            current_image["annotated"] and current_image["mask_path"].exists()
        )
        
        if has_current_annotation or has_saved_annotation:
            # Create ZIP file with mask and JSON
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                if has_current_annotation:
                    # Use current canvas data
                    mask_data = canvas_result.image_data[:, :, 3]
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
                else:
                    # Use saved annotation
                    existing_mask = cv2.imread(
                        str(current_image["mask_path"]), cv2.IMREAD_GRAYSCALE
                    )
                    _, mask_buffer = cv2.imencode(".png", existing_mask)
                    mask_bytes = mask_buffer.tobytes()
                    
                    if current_image["metadata_path"].exists():
                        with open(current_image["metadata_path"], "r") as f:
                            existing_json = json.load(f)
                        json_bytes = json.dumps(existing_json, indent=2).encode("utf-8")
                    else:
                        json_bytes = b"{}"
                
                # Add files to ZIP
                zf.writestr(f"{file_id}_mask.png", mask_bytes)
                zf.writestr(f"{file_id}_annotation.json", json_bytes)
            
            zip_buffer.seek(0)
            
            st.download_button(
                label="📦 Descargar Todo (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"{file_id}_annotation.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary",
            )
            
            st.caption(f"Contenido: `{file_id}_mask.png` + `{file_id}_annotation.json`")
        else:
            st.info("Dibuje una anotación para habilitar la descarga")


if __name__ == "__main__":
    main()
