"""
🐾 Animals Detection — Multi-Phone Live Feed
=============================================
Supports up to 2 phone cameras via IP Webcam / DroidCam / RTSP.

Run:  streamlit run multi_cam_animals.py
"""

import os
# Silencing FFmpeg and OpenCV logs at the OS level
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

import streamlit as st
import cv2
import numpy as np
import threading
import time
import json
import logging
import urllib.request
from datetime import datetime
from pathlib import Path

# ── Logging Filter ─────────────────────────────────────────────────────────────
class _SilentFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        # Suppress common FFmpeg/OpenCV noise if it leaks into Python logs
        suppress = ["Stream ends prematurely", "should be 18446", "unexpected end of file"]
        return not any(s in msg for s in suppress)

logging.getLogger().addFilter(_SilentFilter())

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🐾 Multi-Cam Animal Detector",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: #080c14;
    color: #c8d8f0;
}
section[data-testid="stSidebar"] {
    background: #0a0e1a;
    border-right: 1px solid #1a2a4a;
}
section[data-testid="stSidebar"] * { color: #8aaad0 !important; }

.hero { font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:800;
        color:#4d9fff; letter-spacing:-2px; line-height:1; }
.sub  { font-size:0.72rem; color:#2a4a7a; letter-spacing:4px;
        text-transform:uppercase; margin-top:4px; }

.cam-card {
    background: #0d1520;
    border: 1px solid #1a2a4a;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 8px;
    position: relative;
}
.cam-card.active { border-color: #4d9fff; box-shadow: 0 0 16px #4d9fff33; }
.cam-label { font-family:'Syne',sans-serif; font-size:0.8rem;
             color:#4d9fff; letter-spacing:2px; margin-bottom:4px; }
.cam-url   { font-size:0.65rem; color:#2a4a7a; word-break:break-all; }
.status-ok  { color:#4dff88; font-size:0.7rem; }
.status-bad { color:#ff4d4d; font-size:0.7rem; }

.chip {
    display:inline-block; background:#0d1a30; border:1px solid #1a3060;
    border-radius:4px; padding:3px 10px; font-size:0.68rem;
    color:#4d9fff; letter-spacing:2px; margin-right:6px;
}
.chip.green { color:#4dff88; border-color:#1a4030; background:#0d1a20; }
.chip.red   { color:#ff6060; border-color:#401a1a; background:#1a0d0d; }

.det-row {
    background:#0d1520; border-left:3px solid #4d9fff;
    border-radius:4px; padding:8px 12px; margin-bottom:6px;
    font-size:0.78rem;
}
.det-row span { color:#4d9fff; font-family:'Syne',sans-serif; }
.det-row small { color:#2a4a7a; float:right; }

div.stButton > button {
    background:#4d9fff; color:#080c14; border:none; border-radius:4px;
    font-family:'Syne',sans-serif; font-weight:700; letter-spacing:1px;
    padding:0.5rem 1.2rem; transition:all .2s;
}
div.stButton > button:hover { background:#7dbfff; box-shadow:0 0 16px #4d9fff55; }

#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
ANIMAL_CLASSES = {
    14:"bird", 15:"cat", 16:"dog", 17:"horse", 18:"sheep",
    19:"cow",  20:"elephant", 21:"bear", 22:"zebra", 23:"giraffe",
}
DATA_DIR = Path("collected_data")
DATA_DIR.mkdir(exist_ok=True)

# ── Session state defaults ─────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "cameras": [
            {"name": "Phone 1", "url": "http://192.168.1.101:8080/video", "enabled": False},
            {"name": "Phone 2", "url": "http://192.168.1.102:8080/video", "enabled": False},
        ],
        "active_cam":   0,
        "running":      False,
        "collect":      False,
        "frame_count":  0,
        "collect_count":0,
        "last_dets":    [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO26…")
def load_model(name):
    from ultralytics import YOLO
    return YOLO(name)

# ── Frame reader (threaded) ────────────────────────────────────────────────────
class CameraReader:
    """Non-blocking threaded camera reader with dual mode (Stream/Snapshot)."""
    def __init__(self, url: str):
        self.url   = url
        self.frame = None
        self.ok    = False
        self.lock  = threading.Lock()
        self._stop = threading.Event()
        
        # Determine mode based on URL
        low_url = url.lower()
        if any(x in low_url for x in [".jpg", ".jpeg", "/shot.jpg", "/jpeg"]):
            self.mode = "snapshot"
        else:
            self.mode = "stream"
            
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop.is_set():
            if self.mode == "snapshot":
                try:
                    # Bypasses FFmpeg log spam by pulling JPEGs directly
                    resp = urllib.request.urlopen(self.url, timeout=2.0)
                    data = resp.read()
                    arr  = np.frombuffer(data, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self.lock:
                            self.frame = frame
                            self.ok = True
                    else:
                        with self.lock: self.ok = False
                except Exception:
                    with self.lock: self.ok = False
                time.sleep(0.05) # Refresh rate for snapshots (~20 FPS)
            else:
                # Video stream mode using OpenCV/FFmpeg
                cap = cv2.VideoCapture(self.url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                while not self._stop.is_set():
                    ret, frame = cap.read()
                    with self.lock:
                        self.ok = ret
                        if ret:
                            self.frame = frame
                    if not ret:
                        break # break inner to reconnect
                    time.sleep(0.01)
                cap.release()
                if not self._stop.is_set():
                    time.sleep(2.0) # wait before retry

    def read(self):
        with self.lock:
            return self.ok, self.frame

    def stop(self):
        self._stop.set()

# ── Detection helper ───────────────────────────────────────────────────────────
def detect(model, frame, conf, iou, animals_only, show_lbl, show_conf, track):
    kwargs = dict(conf=conf, iou=iou, verbose=False)
    if track:
        results = model.track(frame, persist=True, **kwargs)
    else:
        results = model.predict(frame, **kwargs)

    result    = results[0]
    annotated = frame.copy()
    dets      = []

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf_v = float(box.conf[0])
            if animals_only and cls_id not in ANIMAL_CLASSES:
                continue
            label = ANIMAL_CLASSES.get(cls_id, result.names.get(cls_id, str(cls_id)))
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1,y1),(x2,y2),(77,159,255),2)
            if show_lbl or show_conf:
                parts = []
                if show_lbl:  parts.append(label.upper())
                if show_conf: parts.append(f"{conf_v:.0%}")
                txt = " ".join(parts)
                (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
                cv2.rectangle(annotated,(x1,y1-th-10),(x1+tw+8,y1),(20,60,120),-1)
                cv2.putText(annotated,txt,(x1+4,y1-5),cv2.FONT_HERSHEY_DUPLEX,0.52,(180,220,255),1)
            dets.append({"label":label,"conf":conf_v,"bbox":(x1,y1,x2,y2)})

    return annotated, dets

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    model_name   = st.selectbox("Model", ["yolo26n.pt","yolo26s.pt","yolo26m.pt","yolo26l.pt"])
    conf_thr     = st.slider("Confidence", 0.10, 0.95, 0.40, 0.05)
    iou_thr      = st.slider("IoU (NMS)",  0.10, 0.95, 0.45, 0.05)

    st.markdown("---")
    animals_only = st.toggle("🐾 Animals Only", True)
    show_lbl     = st.toggle("Show Labels",     True)
    show_conf_v  = st.toggle("Show Confidence", True)
    enable_track = st.toggle("Enable Tracking", False)

    st.markdown("---")
    st.markdown("### 📷 Camera Setup")
    st.caption("TIP: Use `.jpg` URL for best stability & zero FFmpeg warnings.")

    for i, cam in enumerate(st.session_state.cameras):
        with st.expander(f"{cam['name']} {'🟢' if cam['enabled'] else '⚫'}"):
            cam["name"]    = st.text_input("Name", cam["name"], key=f"cn_{i}")
            cam["url"]     = st.text_input("URL", cam["url"], key=f"cu_{i}")
            cam["enabled"] = st.checkbox("Enabled", cam["enabled"], key=f"ce_{i}")

    st.markdown("---")
    st.markdown("### 💾 Data Collection")
    save_dir = st.text_input("Save folder", str(DATA_DIR))
    save_raw  = st.checkbox("Save raw frames",       True)
    save_ann  = st.checkbox("Save annotated frames", True)
    save_json = st.checkbox("Save detection JSON",   True)
    every_n   = st.number_input("Collect every N frames", 1, 60, 5)

    if st.button("🗑 Clear collected data"):
        import shutil
        shutil.rmtree(save_dir, ignore_errors=True)
        Path(save_dir).mkdir(exist_ok=True)
        st.session_state.collect_count = 0
        st.success("Cleared.")

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.65rem;color:#1a3060;line-height:1.8">'
        "MULTI-CAM · YOLO26<br>Snapshot & Stream Modes<br>Silence Fix v1.2"
        "</div>", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([3,1])
with c1:
    st.markdown('<p class="hero">MULTI-CAM<br>ANIMAL DETECTOR</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub">2-Phone Live Feed · YOLO26 · Advanced Resilience</p>', unsafe_allow_html=True)
with c2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<span class="chip green">YOLO26</span>',
        unsafe_allow_html=True)

st.markdown("---")

# ── Camera switcher UI ─────────────────────────────────────────────────────────
st.markdown("#### 🔄 Active Camera")
cam_cols = st.columns(2)
for i, (col, cam) in enumerate(zip(cam_cols, st.session_state.cameras)):
    with col:
        active = st.session_state.active_cam == i
        border = "border:2px solid #4d9fff;" if active else "border:1px solid #1a2a4a;"
        enabled_txt = "🟢 ON" if cam["enabled"] else "⚫ OFF"
        st.markdown(
            f'<div class="cam-card {"active" if active else ""}" style="{border}">'
            f'<div class="cam-label">{cam["name"]}</div>'
            f'<div class="cam-url" style="font-size:0.55rem">{cam["url"]}</div>'
            f'<div class="{"status-ok" if cam["enabled"] else "status-bad"}">{enabled_txt}</div>'
            f'</div>',
            unsafe_allow_html=True)
        if st.button(f"Select →", key=f"sw_{i}", disabled=not cam["enabled"]):
            st.session_state.active_cam = i
            st.rerun()

st.markdown("---")

# ── Control row ────────────────────────────────────────────────────────────────
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 1])
with ctrl1:
    if st.button("▶ Start Detection"):
        st.session_state.running = True
with ctrl2:
    if st.button("⏹ Stop"):
        st.session_state.running = False
with ctrl3:
    if st.button("💾 Start Collecting"):
        st.session_state.collect = True
        Path(save_dir).mkdir(parents=True, exist_ok=True)
with ctrl4:
    if st.button("⏸ Stop Collecting"):
        st.session_state.collect = False

# ── Stats bar ──────────────────────────────────────────────────────────────────
stat_ph = st.empty()

# ── Main live view ─────────────────────────────────────────────────────────────
main_col, side_col = st.columns([3, 1])

with main_col:
    st.markdown("**Live Feed**")
    orig_ph   = st.empty()
    annot_ph  = st.empty()

with side_col:
    st.markdown("**Detections**")
    det_ph    = st.empty()

# ── Thumbnail row (all 4 cams) ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("**All Camera Feeds**")
thumb_cols = st.columns(2)
thumb_phs  = [c.empty() for c in thumb_cols]

# ── Run loop ───────────────────────────────────────────────────────────────────
if st.session_state.running:
    active_idx = st.session_state.active_cam
    active_cam = st.session_state.cameras[active_idx]
    if not active_cam["enabled"]:
        st.error(f"{active_cam['name']} is disabled.")
        st.stop()

    model = load_model(model_name)
    readers = {}
    for i, cam in enumerate(st.session_state.cameras):
        if cam["enabled"]:
            readers[i] = CameraReader(cam["url"])

    time.sleep(1.0)
    frame_idx = 0

    while st.session_state.running:
        reader = readers.get(active_idx)
        if reader is None: break

        ok, frame = reader.read()
        if not ok or frame is None:
            stat_ph.markdown('<span class="chip red">❌ No Feed</span>', unsafe_allow_html=True)
            time.sleep(0.1)
            continue

        t0 = time.perf_counter()
        annotated, dets = detect(model, frame, conf_thr, iou_thr,
                                 animals_only, show_lbl, show_conf_v, enable_track)
        elapsed = (time.perf_counter() - t0) * 1000

        orig_ph.image(cv2.cvtColor(frame,     cv2.COLOR_BGR2RGB), use_container_width=True)
        annot_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        collect_status = f'<span class="chip green">💾 {st.session_state.collect_count} saved</span>' if st.session_state.collect else ""
        stat_ph.markdown(
            f'<span class="chip">📷 {active_cam["name"]}</span>'
            f'<span class="chip">⚡ {elapsed:.0f} ms</span>'
            f'<span class="chip green">🐾 {len(dets)} animals</span>'
            f'{collect_status}',
            unsafe_allow_html=True,
        )

        det_html = ""
        for d in sorted(dets, key=lambda x: -x["conf"]):
            det_html += f'<div class="det-row"><span>🐾 {d["label"].upper()}</span> <small>{d["conf"]:.0%}</small></div>'
        if not dets: det_html = "No animals detected"
        det_ph.markdown(det_html, unsafe_allow_html=True)

        # Thumbnails
        for i, (cam, t_ph) in enumerate(zip(st.session_state.cameras, thumb_phs)):
            r = readers.get(i)
            if r:
                ok_t, f_t = r.read()
                if ok_t and f_t is not None:
                    t_ph.image(cv2.cvtColor(f_t, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Collection
        if st.session_state.collect and (frame_idx % every_n == 0):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(f"{save_dir}/{active_cam['name']}_{ts}_raw.jpg", frame)
            st.session_state.collect_count += 1

        frame_idx += 1
        time.sleep(0.01)

    for r in readers.values(): r.stop()

else:
    for ph in [orig_ph, annot_ph]:
        ph.markdown('<div style="background:#0d1520;height:240px;display:flex;align-items:center;justify-content:center;color:#1a3060">PRESS ▶ START DETECTION</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div style="text-align:center;font-size:0.65rem;color:#0d1a30;letter-spacing:3px">MULTI-CAM ANIMALS DETECTION · YOLO26 · SILENCED FFmpeg</div>', unsafe_allow_html=True)
