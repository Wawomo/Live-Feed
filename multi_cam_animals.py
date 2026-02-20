"""
🐾 Animals Detection — Multi-Phone LIVE Feed
=============================================
TRUE live streaming — no cv2 required.
Decodes MJPEG streams directly using pure Python (urllib + PIL).

Run:   streamlit run multi_cam_animals.py

Phone setup (Android):
  1. Install "IP Webcam" from Play Store (free)
  2. Open app → tap "Start server" at the bottom
  3. All phones + PC must be on the same WiFi
  4. Use the URL shown on the phone screen:
       http://<phone-ip>:8080/video    ← LIVE MJPEG stream ✅
"""

import os, logging, threading, time, json, re
from datetime import datetime
from pathlib import Path
from io import BytesIO
import urllib.request
import urllib.error

os.environ["OPENCV_LOG_LEVEL"]       = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

class _Quiet(logging.Filter):
    _kw = ("Stream ends prematurely", "should be 18446", "moov atom")
    def filter(self, r): return not any(k in r.getMessage() for k in self._kw)
logging.getLogger().addFilter(_Quiet())

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🐾 Multi-Cam Live Detector",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'JetBrains Mono',monospace;background:#080c14;color:#c8d8f0}
section[data-testid="stSidebar"]{background:#0a0e1a;border-right:1px solid #1a2a4a}
section[data-testid="stSidebar"] *{color:#8aaad0 !important}
.hero{font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;color:#4d9fff;letter-spacing:-2px;line-height:1.05}
.sub{font-size:.72rem;color:#2a4a7a;letter-spacing:4px;text-transform:uppercase;margin-top:6px}
.cam-card{background:#0d1520;border:1px solid #1a2a4a;border-radius:8px;padding:10px;margin-bottom:8px}
.cam-card.active{border:2px solid #4d9fff !important;box-shadow:0 0 18px #4d9fff33}
.cam-label{font-family:'Syne',sans-serif;font-size:.82rem;color:#4d9fff;letter-spacing:2px;margin-bottom:4px}
.cam-url{font-size:.62rem;color:#2a4a7a;word-break:break-all}
.chip{display:inline-block;background:#0d1a30;border:1px solid #1a3060;border-radius:4px;padding:3px 10px;font-size:.68rem;color:#4d9fff;letter-spacing:2px;margin-right:6px}
.chip.g{color:#4dff88;border-color:#1a4030;background:#0d1a20}
.chip.r{color:#ff6060;border-color:#401a1a;background:#1a0d0d}
.chip.y{color:#ffcc44;border-color:#403010;background:#1a1400}
.det-row{background:#0d1520;border-left:3px solid #4d9fff;border-radius:4px;padding:8px 12px;margin-bottom:6px;font-size:.78rem}
.det-row .lbl{color:#4d9fff;font-family:'Syne',sans-serif}
.det-row .pct{color:#2a4a7a;float:right}
div.stButton>button{background:#4d9fff;color:#080c14;border:none;border-radius:4px;font-family:'Syne',sans-serif;font-weight:700;padding:.5rem 1.1rem;transition:all .2s}
div.stButton>button:hover{background:#80c8ff;box-shadow:0 0 14px #4d9fff55}
#MainMenu,footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
ANIMAL_CLASSES = {
    14:"bird", 15:"cat", 16:"dog", 17:"horse", 18:"sheep",
    19:"cow",  20:"elephant", 21:"bear", 22:"zebra", 23:"giraffe",
}
DATA_DIR = Path("collected_data")
DATA_DIR.mkdir(exist_ok=True)

# ── Session state ──────────────────────────────────────────────────────────────
_DEFAULTS = {
    "cameras": [
        {"name":"Phone 1","url":"http://192.168.1.101:8080/video","enabled":True},
        {"name":"Phone 2","url":"http://192.168.1.102:8080/video","enabled":False},
    ],
    "active_cam":0,"running":False,"collect":False,
    "collect_count":0,"frame_count":0,"last_dets":[],
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  PURE-PYTHON MJPEG STREAM READER
#  Reads the multipart/x-mixed-replace HTTP stream frame by frame.
#  Works on Streamlit Cloud — zero native dependencies.
# ══════════════════════════════════════════════════════════════════════════════
class MJPEGReader:
    """
    Reads a live MJPEG stream (e.g. IP Webcam /video endpoint) in a
    background thread using only Python's stdlib + Pillow.

    Protocol:
      HTTP response is multipart/x-mixed-replace with boundary markers.
      Each part looks like:
        --<boundary>\r\n
        Content-Type: image/jpeg\r\n
        Content-Length: <N>\r\n
        \r\n
        <N bytes of JPEG data>\r\n
    """

    # Matches "Content-Length: 12345"
    _CL_RE = re.compile(rb"Content-Length:\s*(\d+)", re.IGNORECASE)

    def __init__(self, url: str, timeout: int = 10):
        self.url     = url
        self.timeout = timeout
        self.frame:  Image.Image | None = None
        self.ok      = False
        self.fps     = 0.0
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    # ── Background thread ──────────────────────────────────────────────────────
    def _run(self):
        while not self._stop.is_set():
            try:
                self._stream()
            except Exception:
                with self._lock:
                    self.ok = False
                time.sleep(2)

    def _stream(self):
        req  = urllib.request.Request(
            self.url,
            headers={"User-Agent": "AnimalDetector/1.0"}
        )
        resp = urllib.request.urlopen(req, timeout=self.timeout)

        buf     = b""
        t_last  = time.perf_counter()

        while not self._stop.is_set():
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk

            while True:
                m = self._CL_RE.search(buf)
                if not m:
                    break

                cl = int(m.group(1))
                header_end = buf.find(b"\r\n\r\n", m.start())
                if header_end == -1:
                    break
                data_start = header_end + 4

                if len(buf) < data_start + cl:
                    break

                jpeg_bytes = buf[data_start : data_start + cl]
                buf        = buf[data_start + cl:]

                try:
                    img = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
                except Exception:
                    continue

                now    = time.perf_counter()
                fps    = 1.0 / max(now - t_last, 1e-6)
                t_last = now

                with self._lock:
                    self.frame = img
                    self.ok    = True
                    self.fps   = fps

    # ── Public API ─────────────────────────────────────────────────────────────
    def read(self) -> tuple[bool, Image.Image | None]:
        with self._lock:
            return self.ok, self.frame.copy() if self.frame else None

    def stop(self):
        self._stop.set()


# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading YOLO26 model…")
def load_model(name: str):
    from ultralytics import YOLO
    return YOLO(name)

# ── PIL bounding-box drawing ───────────────────────────────────────────────────
def draw_boxes(img: Image.Image, dets: list,
               show_lbl: bool, show_conf: bool) -> Image.Image:
    out  = img.copy()
    draw = ImageDraw.Draw(out)
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(77, 159, 255), width=2)
        if show_lbl or show_conf:
            parts = []
            if show_lbl:  parts.append(d["label"].upper())
            if show_conf: parts.append(f"{d['conf']:.0%}")
            txt = "  ".join(parts)
            tw  = int(draw.textlength(txt)) + 10
            draw.rectangle([x1, y1 - 20, x1 + tw, y1], fill=(20, 60, 120))
            draw.text((x1 + 5, y1 - 17), txt, fill=(180, 220, 255))
    return out

# ── YOLO detection ─────────────────────────────────────────────────────────────
def run_detect(model, pil_img, conf, iou, animals_only, show_lbl, show_conf, track):
    arr = np.array(pil_img)
    kw  = dict(conf=conf, iou=iou, verbose=False)
    res = model.track(arr, persist=True, **kw) if track else model.predict(arr, **kw)
    result = res[0]
    dets   = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf_v = float(box.conf[0])
            if animals_only and cls_id not in ANIMAL_CLASSES:
                continue
            label = ANIMAL_CLASSES.get(cls_id, result.names.get(cls_id, str(cls_id)))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dets.append({"label": label, "conf": conf_v, "bbox": (x1, y1, x2, y2)})
    annotated = draw_boxes(pil_img, dets, show_lbl, show_conf)
    return annotated, dets

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    model_name  = st.selectbox("Model", ["yolo26n.pt","yolo26s.pt","yolo26m.pt","yolo26l.pt"])
    conf_thr    = st.slider("Confidence", 0.10, 0.95, 0.40, 0.05)
    iou_thr     = st.slider("IoU (NMS)",  0.10, 0.95, 0.45, 0.05)

    st.markdown("---")
    animals_only = st.toggle("🐾 Animals Only",   True)
    show_lbl     = st.toggle("Show Labels",        True)
    show_conf_v  = st.toggle("Show Confidence",    True)
    enable_track = st.toggle("Enable Tracking",    False)

    st.markdown("---")
    st.markdown("### 📷 Camera URLs")
    st.caption(
        "**IP Webcam** (Android) → Start server  \n"
        "Use `/video` for true live MJPEG stream:  \n"
        "`http://192.168.x.x:8080/video`"
    )
    for i, cam in enumerate(st.session_state.cameras):
        with st.expander(f"{cam['name']} {'🟢' if cam['enabled'] else '⚫'}"):
            cam["name"]    = st.text_input("Name",    cam["name"],    key=f"cn{i}")
            cam["url"]     = st.text_input("URL",     cam["url"],     key=f"cu{i}")
            cam["enabled"] = st.checkbox("Enabled",   cam["enabled"], key=f"ce{i}")
            base = cam["url"].rsplit("/", 1)[0]
            st.caption(
                f"📡 Live stream: `{base}/video`  \n"
                f"📸 Snapshot:    `{base}/shot.jpg`"
            )

    st.markdown("---")
    st.markdown("### 💾 Data Collection")
    save_dir  = st.text_input("Save folder", str(DATA_DIR))
    save_raw  = st.checkbox("Save raw frames",       True)
    save_ann  = st.checkbox("Save annotated frames", True)
    save_meta = st.checkbox("Save detection JSON",   True)
    every_n   = st.number_input("Every N frames", 1, 120, 10)
    if st.button("🗑 Clear data"):
        import shutil
        shutil.rmtree(save_dir, ignore_errors=True)
        Path(save_dir).mkdir(exist_ok=True)
        st.session_state.collect_count = 0
        st.success("Cleared.")

    st.markdown("---")
    st.markdown(
        '<div style="font-size:.65rem;color:#1a3060;line-height:2">'
        'MJPEG stream · Pure Python<br>'
        'Pillow · urllib · numpy<br>'
        'No cv2 required ✅'
        '</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown('<p class="hero">MULTI-CAM<br>LIVE DETECTOR</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub">2-Phone MJPEG Stream · YOLO26 · Pure Python · No cv2</p>',
        unsafe_allow_html=True)
with h2:
    n_en = sum(1 for c in st.session_state.cameras if c["enabled"])
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<span class="chip">{n_en}/2 LIVE</span>'
        f'<span class="chip g">MJPEG ✅</span>',
        unsafe_allow_html=True)

st.markdown("---")

# ── Camera switcher ────────────────────────────────────────────────────────────
st.markdown("#### 🔄 Select Active Camera")
cam_cols = st.columns(2)
for i, (col, cam) in enumerate(zip(cam_cols, st.session_state.cameras)):
    with col:
        active = st.session_state.active_cam == i
        cls    = "cam-card active" if active else "cam-card"
        st.markdown(
            f'<div class="{cls}">'
            f'<div class="cam-label">{"▶ " if active else ""}{cam["name"]}</div>'
            f'<div class="cam-url">{cam["url"]}</div>'
            f'<div style="color:{"#4dff88" if cam["enabled"] else "#ff4d4d"};font-size:.7rem">'
            f'{"🟢 LIVE" if cam["enabled"] else "⚫ OFF"}</div>'
            f'</div>', unsafe_allow_html=True)
        if st.button("Select", key=f"sw{i}", disabled=not cam["enabled"]):
            st.session_state.active_cam = i
            st.rerun()

st.markdown("---")

# ── Controls ───────────────────────────────────────────────────────────────────
b1, b2, b3, b4 = st.columns(4)
with b1:
    if st.button("▶ Start"):
        st.session_state.running = True
with b2:
    if st.button("⏹ Stop"):
        st.session_state.running = False
with b3:
    if st.button("💾 Collect ON"):
        st.session_state.collect = True
        Path(save_dir).mkdir(parents=True, exist_ok=True)
with b4:
    if st.button("⏸ Collect OFF"):
        st.session_state.collect = False

stat_ph = st.empty()

left_col, right_col = st.columns([3, 1])
with left_col:
    st.markdown("**📡 Live Feed**")
    orig_ph  = st.empty()
    annot_ph = st.empty()
with right_col:
    st.markdown("**🐾 Detections**")
    det_ph  = st.empty()
    fps_ph  = st.empty()

st.markdown("---")
st.markdown("**All Feeds**")
th_cols = st.columns(2)
th_phs  = [c.empty() for c in th_cols]
for col, cam in zip(th_cols, st.session_state.cameras):
    col.caption(cam["name"])

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LIVE LOOP
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.running:
    idx     = st.session_state.active_cam
    act_cam = st.session_state.cameras[idx]

    if not act_cam["enabled"]:
        st.error(f"**{act_cam['name']}** is disabled — enable it in the sidebar.")
        st.stop()

    model = load_model(model_name)

    # Start MJPEG readers for all enabled cameras
    readers: dict[int, MJPEGReader] = {
        i: MJPEGReader(cam["url"])
        for i, cam in enumerate(st.session_state.cameras)
        if cam["enabled"]
    }

    with st.spinner("📡 Connecting to camera streams…"):
        time.sleep(2.0)

    frame_idx      = 0
    save_path      = Path(save_dir)
    no_frame_count = 0

    while st.session_state.running:
        reader = readers.get(idx)
        if reader is None:
            break

        ok, pil_img = reader.read()

        # ── No frame yet ──
        if not ok or pil_img is None:
            no_frame_count += 1
            stat_ph.markdown(
                f'<span class="chip r">❌ {act_cam["name"]} — waiting for stream…</span>'
                f'<span class="chip y">Check: same WiFi? IP correct? Server running?</span>',
                unsafe_allow_html=True)
            if no_frame_count > 50:
                st.error(
                    f"**Cannot connect to {act_cam['name']}**  \n\n"
                    f"URL tried: `{act_cam['url']}`  \n\n"
                    "**Checklist:**  \n"
                    "- Phone and PC on same WiFi?  \n"
                    "- IP Webcam server running on phone?  \n"
                    "- Try opening the URL in your browser first  \n"
                    "- Firewall blocking port 8080?"
                )
            time.sleep(0.2)
            continue

        no_frame_count = 0

        # ── Run YOLO ──
        t0 = time.perf_counter()
        annotated, dets = run_detect(
            model, pil_img, conf_thr, iou_thr,
            animals_only, show_lbl, show_conf_v, enable_track,
        )
        infer_ms = (time.perf_counter() - t0) * 1000

        # ── Display ──
        orig_ph.image(pil_img,    caption="Original",  use_container_width=True)
        annot_ph.image(annotated,  caption="Detected",  use_container_width=True)

        # ── Stats ──
        stream_fps = reader.fps
        coll_chip  = (
            f'<span class="chip g">💾 {st.session_state.collect_count} saved</span>'
            if st.session_state.collect else '<span class="chip">💾 off</span>'
        )
        stat_ph.markdown(
            f'<span class="chip">📡 {act_cam["name"]}</span>'
            f'<span class="chip g">{stream_fps:.1f} fps</span>'
            f'<span class="chip">⚡ {infer_ms:.0f} ms</span>'
            f'<span class="chip g">🐾 {len(dets)}</span>'
            f'<span class="chip">Frame {frame_idx}</span>'
            f'{coll_chip}',
            unsafe_allow_html=True,
        )

        # ── Detection cards ──
        if dets:
            det_html = "".join(
                f'<div class="det-row">'
                f'<span class="lbl">🐾 {d["label"].upper()}</span>'
                f'<span class="pct">{d["conf"]:.0%}</span><br>'
                f'<span style="color:#1a3a60;font-size:.62rem">'
                f'{"█" * int(d["conf"] * 10)}</span>'
                f'</div>'
                for d in sorted(dets, key=lambda x: -x["conf"])
            )
        else:
            det_html = (
                '<div style="color:#2a4a7a;font-size:.78rem;padding:10px 0">'
                'No animals detected<br>'
                '<span style="font-size:.62rem">↓ lower confidence?</span>'
                '</div>'
            )
        det_ph.markdown(det_html, unsafe_allow_html=True)
        fps_ph.markdown(
            f'<div style="font-size:.62rem;color:#1a3060;margin-top:8px">'
            f'Stream: {stream_fps:.1f} fps<br>'
            f'Infer:  {infer_ms:.0f} ms</div>',
            unsafe_allow_html=True)

        # ── Thumbnail strip ──
        for i, (cam, t_ph) in enumerate(zip(st.session_state.cameras, th_phs)):
            if not cam["enabled"]:
                t_ph.markdown(
                    '<div style="background:#0d1520;border:1px solid #1a2a4a;height:70px;'
                    'border-radius:6px;display:flex;align-items:center;justify-content:center;'
                    'color:#1a3060;font-size:.6rem">DISABLED</div>',
                    unsafe_allow_html=True)
                continue
            r = readers.get(i)
            if r:
                ok_t, f_t = r.read()
                if ok_t and f_t:
                    t_ph.image(f_t, use_container_width=True)

        # ── Data collection ──
        if st.session_state.collect and frame_idx % every_n == 0:
            ts     = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            tag    = act_cam["name"].replace(" ", "_")
            prefix = save_path / f"{tag}_{ts}"
            if save_raw:  pil_img.save(f"{prefix}_raw.jpg",  quality=92)
            if save_ann:  annotated.save(f"{prefix}_ann.jpg", quality=92)
            if save_meta and dets:
                with open(f"{prefix}_dets.json", "w") as jf:
                    json.dump({
                        "camera": act_cam["name"], "url": act_cam["url"],
                        "timestamp": ts, "model": model_name, "frame": frame_idx,
                        "stream_fps": round(stream_fps, 2),
                        "detections": [
                            {"label": d["label"], "confidence": round(d["conf"], 4),
                             "bbox": list(d["bbox"])} for d in dets
                        ],
                    }, jf, indent=2)
            st.session_state.collect_count += 1

        frame_idx += 1
        st.session_state.frame_count = frame_idx

    for r in readers.values():
        r.stop()

else:
    _IDLE = (
        '<div style="background:#0d1520;border:1px dashed #1a2a4a;border-radius:8px;'
        'height:180px;display:flex;align-items:center;justify-content:center;'
        'color:#1a3060;font-size:.85rem;letter-spacing:3px">{}</div>'
    )
    orig_ph.markdown(_IDLE.format("ORIGINAL FEED"), unsafe_allow_html=True)
    annot_ph.markdown(_IDLE.format("▶ PRESS START"), unsafe_allow_html=True)
    det_ph.markdown(
        '<div style="color:#1a3060;font-size:.75rem;padding:8px">Waiting…</div>',
        unsafe_allow_html=True)
    if st.session_state.collect_count > 0:
        st.success(f"💾 {st.session_state.collect_count} frames in `{save_dir}`")

# ── Data preview ───────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📁 Collected Data Preview", expanded=False):
    imgs = sorted(Path(save_dir).glob("*_ann.jpg"))[-8:]
    if imgs:
        cols = st.columns(min(4, len(imgs)))
        for col, p in zip(cols * 2, imgs):
            col.image(str(p), caption=p.stem[-18:], use_container_width=True)
    else:
        st.info("No collected frames yet — start detection and enable collecting.")
    jsons = sorted(Path(save_dir).glob("*_dets.json"))
    if jsons:
        st.markdown(f"**{len(jsons)} detection JSON files** — latest:")
        with open(jsons[-1]) as f:
            st.json(json.load(f))

st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:.62rem;color:#0d1a30;letter-spacing:3px">'
    'MULTI-CAM LIVE · YOLO26 · MJPEG · PURE PYTHON · NO CV2'
    '</div>', unsafe_allow_html=True)
