
"""
🐾 Animals Detection using Ultralytics YOLO26 + Streamlit
=========================================================
Run with:  streamlit run animals_detection.py

Requirements:
    pip install ultralytics streamlit opencv-python-headless Pillow
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🐾 Animal Detector",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Mono', monospace;
        background-color: #0a0f0a;
        color: #d4e8d4;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d160d;
        border-right: 1px solid #1e3a1e;
    }
    section[data-testid="stSidebar"] * { color: #a8cba8 !important; }

    /* Hero title */
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -2px;
        color: #7dff7d;
        text-shadow: 0 0 40px #3aff3a55;
        margin: 0;
        line-height: 1;
    }
    .hero-sub {
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        color: #4a7a4a;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-top: 6px;
    }

    /* Stats chips */
    .chip {
        display: inline-block;
        background: #0f2a0f;
        border: 1px solid #2a5a2a;
        border-radius: 4px;
        padding: 4px 12px;
        font-size: 0.72rem;
        color: #7dff7d;
        letter-spacing: 2px;
        margin-right: 8px;
    }

    /* Detection card */
    .det-card {
        background: linear-gradient(135deg, #0f2a0f 0%, #0a1a0a 100%);
        border: 1px solid #1e3a1e;
        border-left: 3px solid #7dff7d;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 8px;
        font-size: 0.82rem;
    }
    .det-label { color: #7dff7d; font-weight: 500; font-family: 'Syne', sans-serif; }
    .det-conf  { color: #4a7a4a; float: right; }

    /* Buttons */
    div.stButton > button {
        background: #7dff7d;
        color: #0a0f0a;
        border: none;
        border-radius: 4px;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 0.55rem 1.4rem;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background: #a0ffb0;
        box-shadow: 0 0 20px #7dff7d55;
    }

    /* Sliders & selects accent */
    .stSlider > div > div > div > div { background: #7dff7d !important; }

    /* Image containers */
    .stImage img { border-radius: 6px; border: 1px solid #1e3a1e; }

    /* Divider */
    hr { border-color: #1e3a1e !important; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── COCO animal class IDs & names ──────────────────────────────────────────────
ANIMAL_CLASSES = {
    0: "person",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    model_choice = st.selectbox(
        "Model",
        ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"],
        help="Larger = more accurate, slower",
    )

    conf_threshold = st.slider("Confidence Threshold", 0.10, 0.95, 0.40, 0.05)
    iou_threshold  = st.slider("IoU (NMS) Threshold",  0.10, 0.95, 0.45, 0.05)

    st.markdown("---")
    animals_only = st.toggle("🐾 Animals Only", value=True,
                             help="Filter to COCO animal classes")
    show_labels  = st.toggle("Show Labels",     value=True)
    show_conf    = st.toggle("Show Confidence", value=True)
    enable_track = st.toggle("Enable Tracking", value=False)

    st.markdown("---")
    source_type = st.radio("Input Source", ["📷 Webcam", "🖼️ Upload Image", "🎬 Upload Video"])

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.7rem;color:#2a5a2a;line-height:1.6">'
        "YOLO26 · Ultralytics<br>"
        "Streamlit Live Inference<br>"
        "Animals Detection Demo"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Header ─────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<p class="hero-title">ANIMAL<br>DETECTOR</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Powered by Ultralytics YOLO26 &nbsp;·&nbsp; Real-time CV</p>',
                unsafe_allow_html=True)
with col_h2:
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="chip">MODEL {model_choice.upper()}</span>'
        f'<span class="chip">CONF {conf_threshold}</span>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Lazy-load model ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO26 model…")
def load_model(model_name: str):
    from ultralytics import YOLO
    return YOLO(model_name)

# ── Detection helper ───────────────────────────────────────────────────────────
def run_detection(model, frame_bgr: np.ndarray, conf: float, iou: float,
                  animals_only: bool, show_labels: bool, show_conf: bool,
                  track: bool) -> tuple[np.ndarray, list]:
    """Run inference on a single BGR frame; return annotated frame + detections."""
    kwargs = dict(conf=conf, iou=iou, verbose=False)

    if track:
        results = model.track(frame_bgr, persist=True, **kwargs)
    else:
        results = model.predict(frame_bgr, **kwargs)

    result   = results[0]
    annotated = frame_bgr.copy()
    detections = []

    boxes = result.boxes
    if boxes is not None and len(boxes):
        for box in boxes:
            cls_id = int(box.cls[0])
            conf_v = float(box.conf[0])

            if animals_only and cls_id not in ANIMAL_CLASSES:
                continue

            label_name = ANIMAL_CLASSES.get(cls_id, result.names.get(cls_id, str(cls_id)))
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (125, 255, 125), 2)

            # Draw label
            if show_labels or show_conf:
                parts = []
                if show_labels: parts.append(label_name.upper())
                if show_conf:   parts.append(f"{conf_v:.0%}")
                text = " ".join(parts)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), (60, 160, 60), -1)
                cv2.putText(annotated, text, (x1 + 4, y1 - 5),
                            cv2.FONT_HERSHEY_DUPLEX, 0.55, (10, 30, 10), 1)

            detections.append({"label": label_name, "conf": conf_v,
                                "bbox": (x1, y1, x2, y2)})

    return annotated, detections

# ── Detection panel helper ─────────────────────────────────────────────────────
def render_detections(detections: list, elapsed_ms: float):
    st.markdown(
        f'<span class="chip">⚡ {elapsed_ms:.0f} ms</span>'
        f'<span class="chip">🐾 {len(detections)} found</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    if not detections:
        st.info("No animals detected — try adjusting the confidence threshold.")
        return
    for d in sorted(detections, key=lambda x: -x["conf"]):
        bar = "█" * int(d["conf"] * 10)
        st.markdown(
            f'<div class="det-card">'
            f'<span class="det-label">🐾 {d["label"].upper()}</span>'
            f'<span class="det-conf">{d["conf"]:.1%}</span>'
            f'<br><span style="color:#2a5a2a;font-size:0.7rem">{bar}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# SOURCE: UPLOAD IMAGE
# ══════════════════════════════════════════════════════════════════════════════
if "🖼️ Upload Image" in source_type:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if uploaded:
        model = load_model(model_choice)
        img   = Image.open(uploaded).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        t0 = time.perf_counter()
        annotated, dets = run_detection(
            model, frame, conf_threshold, iou_threshold,
            animals_only, show_labels, show_conf, False,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original**")
            st.image(img, use_container_width=True)
        with col2:
            st.markdown("**Detected**")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("---")
        render_detections(dets, elapsed)
    else:
        st.info("⬆️  Upload an image to begin detection.")

# ══════════════════════════════════════════════════════════════════════════════
# SOURCE: UPLOAD VIDEO
# ══════════════════════════════════════════════════════════════════════════════
elif "🎬 Upload Video" in source_type:
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        model = load_model(model_choice)

        # Write to tmp file
        tmp_path = "/tmp/upload_video.mp4"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        cap  = cv2.VideoCapture(tmp_path)
        fps  = cap.get(cv2.CAP_PROP_FPS) or 25
        total= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        col1, col2 = st.columns(2)
        with col1:
            orig_ph = st.empty()
        with col2:
            det_ph  = st.empty()

        info_ph = st.empty()
        prog    = st.progress(0)
        stop_btn = st.button("⏹ Stop")

        frame_idx = 0
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            annotated, dets = run_detection(
                model, frame, conf_threshold, iou_threshold,
                animals_only, show_labels, show_conf, enable_track,
            )
            elapsed = (time.perf_counter() - t0) * 1000

            orig_ph.image(cv2.cvtColor(frame,     cv2.COLOR_BGR2RGB), use_container_width=True)
            det_ph.image( cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            pct = int((frame_idx / max(total, 1)) * 100)
            prog.progress(min(pct, 100))
            info_ph.markdown(
                f'<span class="chip">Frame {frame_idx}</span>'
                f'<span class="chip">⚡ {elapsed:.0f} ms</span>'
                f'<span class="chip">🐾 {len(dets)} animals</span>',
                unsafe_allow_html=True,
            )
            frame_idx += 1
            time.sleep(max(0, 1 / fps - elapsed / 1000))

        cap.release()
        st.success("✅ Video processing complete.")
    else:
        st.info("⬆️  Upload a video to begin detection.")

# ══════════════════════════════════════════════════════════════════════════════
# SOURCE: WEBCAM (via Ultralytics solutions.Inference)
# ══════════════════════════════════════════════════════════════════════════════
else:  # Webcam
    st.markdown(
        """
        ### 📷 Webcam Live Inference

        The webcam mode uses the **Ultralytics `solutions.Inference`** module directly —
        the recommended approach from the official docs. Click **Launch** to open the
        Streamlit inference UI in a new tab (or it will embed here if already running
        inside the Streamlit server).
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        launch = st.button("🚀 Launch Live Inference")

    with col_b:
        st.markdown(
            """
            <div style="font-size:0.75rem;color:#4a7a4a;padding-top:8px">
            This runs <code>solutions.Inference(model=…)</code> exactly as documented
            at docs.ultralytics.com/guides/streamlit-live-inference
            </div>
            """,
            unsafe_allow_html=True,
        )

    if launch:
        try:
            from ultralytics import solutions

            st.info("🎥 Starting Ultralytics Streamlit inference — your webcam feed will appear below.")

            inf = solutions.Inference(
                model=model_choice,
            )
            inf.inference()          # Launches the built-in Streamlit component

        except ImportError:
            st.error("Ultralytics not installed. Run:  `pip install ultralytics`")
        except Exception as e:
            st.error(f"Error: {e}")

    # Fallback manual webcam (works when running locally with cv2 camera access)
    with st.expander("🔧 Manual Webcam (fallback — local only)", expanded=False):
        st.markdown("Uses OpenCV directly; no browser camera permission required.")
        cam_idx = st.number_input("Camera index", 0, 10, 0, step=1)

        if st.button("▶ Start Manual Webcam"):
            model = load_model(model_choice)
            cap   = cv2.VideoCapture(int(cam_idx))

            if not cap.isOpened():
                st.error(f"Cannot open camera {cam_idx}")
            else:
                col1, col2 = st.columns(2)
                orig_ph = col1.empty()
                det_ph  = col2.empty()
                info_ph = st.empty()
                stop_ph = st.empty()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("No frame captured.")
                        break

                    t0 = time.perf_counter()
                    annotated, dets = run_detection(
                        model, frame, conf_threshold, iou_threshold,
                        animals_only, show_labels, show_conf, enable_track,
                    )
                    elapsed = (time.perf_counter() - t0) * 1000

                    orig_ph.image(cv2.cvtColor(frame,     cv2.COLOR_BGR2RGB), use_container_width=True)
                    det_ph.image( cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                    info_ph.markdown(
                        f'<span class="chip">⚡ {elapsed:.0f} ms</span>'
                        f'<span class="chip">🐾 {len(dets)} animals</span>',
                        unsafe_allow_html=True,
                    )

                cap.release()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:0.7rem;color:#1e3a1e;letter-spacing:3px">'
    "ANIMALS DETECTION · YOLO26 · ULTRALYTICS · STREAMLIT"
    "</div>",
    unsafe_allow_html=True,
)
