"""
Herramienta de Anotación Ground Truth para Radiólogos - Consolidación por Neumonía

Características:
1. Navegar imágenes de rayos X desde la carpeta Pacientes automáticamente
2. Anotar consolidación directamente en el navegador (sin herramientas externas)
3. Múltiples entradas de consolidación para neumonía multilobar
4. Guardar máscara + JSON de metadatos en la misma carpeta del paciente
5. Seguimiento de progreso, comparación inter-evaluador, zoom, tema oscuro
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
import os
import logging
from datetime import datetime
from streamlit_drawable_canvas import st_canvas
import hashlib

# Minimal logging for debugging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
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
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
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
    base = Path(base_path).resolve()
    patient_images = []
    
    if not base.exists():
        return patient_images

    # Get all subdirectories
    try:
        subdirs = [f for f in base.iterdir() if f.is_dir()]
    except Exception:
        subdirs = []
    
    folders = [base] + subdirs
    
    for folder in folders:
        # Collect all image extensions (case-insensitive)
        img_files = []
        for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
            img_files.extend(list(folder.glob(ext)))
        
        img_files = sorted(set(img_files))
        
        for img in img_files:
            if "_mask" in img.name:
                continue
            mask_path = img.parent / f"{img.stem}_mask.png"
            meta_path = img.parent / f"{img.stem}_annotation.json"
            is_base_folder = folder.resolve() == base
            patient_id = "uploaded" if is_base_folder else folder.name
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
    
    # Create upload directory
    upload_dir = Path("./uploaded_images").resolve()
    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"Error creating upload directory: {e}")
    
    # Handle uploaded files - clear previous images when new ones are uploaded
    if uploaded_files:
        # Get names of currently uploaded files
        uploaded_names = {uf.name for uf in uploaded_files}
        
        # Delete all existing files that are not in the current upload
        try:
            for existing_file in upload_dir.iterdir():
                if existing_file.is_file() and existing_file.name not in uploaded_names:
                    existing_file.unlink()
        except Exception:
            pass
        
        # Save new files
        new_files_uploaded = False
        for uf in uploaded_files:
            file_path = upload_dir / uf.name
            if not file_path.exists():
                try:
                    with open(file_path, "wb") as f:
                        f.write(uf.getbuffer())
                    new_files_uploaded = True
                except Exception as e:
                    st.sidebar.error(f"Error saving {uf.name}: {e}")
        
        if new_files_uploaded:
            # Reset navigation state for new images
            st.session_state.current_index = 0
            st.sidebar.success(f"✅ ¡{len(uploaded_files)} imagen(es) subida(s)!")
            st.rerun()
    
    st.sidebar.divider()

    # ── Load images ────────────────────────────────────────────────────
    patients_path = str(upload_dir)
    patient_images = get_all_patient_images(patients_path)

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

    # ── Sidebar: progress ──────────────────────────────────────────────
    annotated_count, total_count, progress_pct = get_annotation_progress(
        patient_images
    )
    st.sidebar.header("📊 Progreso")
    st.sidebar.progress(progress_pct / 100)
    st.sidebar.metric("Anotadas", f"{annotated_count} / {total_count}")
    st.sidebar.metric("Completado", f"{progress_pct:.1f}%")

    # ── Sidebar: filter ────────────────────────────────────────────────
    st.sidebar.header("🔍 Filtrar")
    show_filter = st.sidebar.radio(
        "Mostrar",
        ["Todas las Imágenes", "Sin Anotar", "Anotadas"],
        index=1,
    )
    if show_filter == "Sin Anotar":
        filtered_images = [i for i in patient_images if not i["annotated"]]
    elif show_filter == "Anotadas":
        filtered_images = [i for i in patient_images if i["annotated"]]
    else:
        filtered_images = patient_images

    if not filtered_images:
        st.warning(f"No hay imágenes que coincidan con el filtro **{show_filter}**.")
        return

    # ── Navigation state ───────────────────────────────────────────────
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if st.session_state.current_index >= len(filtered_images):
        st.session_state.current_index = 0
    current_image = filtered_images[st.session_state.current_index]

    # ── Sidebar: drawing settings ──────────────────────────────────────
    st.sidebar.header("🎨 Configuración de Dibujo")
    stroke_width = st.sidebar.slider("Tamaño del Pincel", 1, 50, 15)

    drawing_mode = st.sidebar.selectbox(
        "Herramienta de Dibujo",
        ["freedraw", "rect", "circle", "line"],
        index=0,
        help="freedraw: pincel libre · rect/circle/line: formas",
    )

    canvas_width = st.sidebar.slider(
        "Ancho del Lienzo (px)", 600, 1400, 900, step=50,
        help="Ajuste para que se adapte a su pantalla. La relación de aspecto siempre se preserva.",
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
        if st.button("➖", key="zoom_out", help="Alejar",
                     use_container_width=True):
            st.session_state.zoom_level = max(
                1.0, round(st.session_state.zoom_level - 0.25, 2)
            )
            st.rerun()
    with zb2:
        if st.button("➕", key="zoom_in", help="Acercar",
                     use_container_width=True):
            st.session_state.zoom_level = min(
                5.0, round(st.session_state.zoom_level + 0.25, 2)
            )
            st.rerun()
    with zb3:
        if st.button("🔄", key="zoom_reset", help="Restablecer zoom",
                     use_container_width=True):
            st.session_state.zoom_level = 1.0
            st.session_state.zoom_pan_x = 0.5
            st.session_state.zoom_pan_y = 0.5
            st.rerun()
    with zb4:
        st.write(f"**{st.session_state.zoom_level:.1f}x**")

    zoom_level = st.sidebar.slider(
        "Nivel de Zoom", 1.0, 5.0, st.session_state.zoom_level, step=0.25,
        help="Arrastre o use los botones ➕/➖ de arriba.",
        key="zoom_slider",
    )
    if zoom_level != st.session_state.zoom_level:
        st.session_state.zoom_level = zoom_level

    if st.session_state.zoom_level > 1.0:
        # Pan controls — arrows + sliders
        st.sidebar.caption("**Desplazar (botones de flecha o deslizadores)**")
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
            "Desplazar H", 0.0, 1.0, st.session_state.zoom_pan_x, step=0.05,
            key="pan_h_slider",
        )
        zoom_pan_y = st.sidebar.slider(
            "Desplazar V", 0.0, 1.0, st.session_state.zoom_pan_y, step=0.05,
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

    # Load original image
    img_rgb = load_image_from_path(current_image["image_path"])
    if img_rgb is None:
        st.error(f"No se puede cargar la imagen: {current_image['image_path']}")
        return

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

        if zoom_level > 1.0:
            st.write(
                f"**🎨 Dibujando con color {active_label}** "
                f"(🔍 {zoom_level:.1f}x — Desplace ↕ para zoom, "
                f"use botones de flecha para desplazar)"
            )
        else:
            st.write(
                f"**🎨 Dibujando con color {active_label}** "
                f"(Desplace ↕ sobre la imagen para zoom)"
            )

        # ONE canvas per image — all sites draw here.
        logger.info(f"Rendering canvas: {canvas_w}x{canvas_h}, image shape: {img_for_canvas.shape}")
        try:
            canvas_result = st_canvas(
                fill_color=fill_rgba,
                stroke_width=stroke_width,
                stroke_color=active_hex,
                background_image=Image.fromarray(img_for_canvas),
                background_color="#000000",
                update_streamlit=True,
                height=canvas_h,
                width=canvas_w,
                drawing_mode=drawing_mode,
                key=f"canvas_{current_image['patient_id']}_"
                    f"{current_image['image_name']}_z{zoom_level}_"
                    f"x{zoom_pan_x}_y{zoom_pan_y}",
            )
            logger.info("Canvas rendered successfully")
        except Exception as e:
            logger.error(f"Canvas render error: {e}")
            st.error(f"Error rendering canvas: {e}")
            canvas_result = None

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
            st.caption("📍 Vista general — el recuadro rojo muestra la región de zoom actual")
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
                    {"location": "Right Lower Lobe",
                     "type": "Solid Consolidation"}
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
