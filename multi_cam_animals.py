"""
🐾 Animals Detection — Multi-Phone Live Feed
=============================================
Works on Streamlit Cloud (Python 3.11+) and locally.

Run:   streamlit run multi_cam_animals.py

Phone setup:
  Android → Install "IP Webcam" (Play Store) → Start Server
  Recommended URL: http://<phone-ip>:8080/shot.jpg
"""

import os, sys, logging, threading, time, json
from datetime import datetime
from pathlib import Path
from io import BytesIO
import urllib.request

# ── Silence FFmpeg noise before cv2 loads ──────────────────────────────────────
os.environ["OPENCV_LOG_LEVEL"]       = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

class _SilentFilter(logging.Filter):
    _kw = ("Stream ends prematurely", "should be 18446", "moov atom", "Invalid data")
    def filter(self, r):
        return not any(k in r.getMessage() for k in self._kw)
logging.getLogger().addFilter(_SilentFilter())

# ── cv2 is optional — falls back to PIL-only snapshot mode ────────────────────
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

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
html,body,[class*="css"]{font-family:'JetBrains Mono',monospace;background:#080c14;color:#c8d8f0}
section[data-testid="stSidebar"]{background:#0a0e1a;border-right:1px solid #1a2a4a}
section[data-testid="stSidebar"] *{color:#8aaad0 !important}
.hero{font-family:'Syne',sans-serif;font-size:2.6rem;font-weight:800;color:#4d9fff;letter-spacing:-2px;line-height:1}
.sub{font-size:.72rem;color:#2a4a7a;letter-spacing:4px;text-transform:uppercase;margin-top:4px}
.cam-card{background:#0d1520;border:1px solid #1a2a4a;border-radius:8px;padding:10px;margin-bottom:8px}
.cam-card.active{border-color:#4d9fff;box-shadow:0 0 16px #4d9fff33}
.cam-label{font-family:'Syne',sans-serif;font-size:.8rem;color:#4d9fff;letter-spacing:2px;margin-bottom:4px}
.cam-url{font-size:.65rem;color:#2a4a7a;word-break:break-all}
.status-ok{color:#4dff88;font-size:.7rem}.status-bad{color:#ff4d4d;font-size:.7rem}
.chip{display:inline-block;background:#0d1a30;border:1px solid #1a3060;border-radius:4px;padding:3px 10px;font-size:.68rem;color:#4d9fff;letter-spacing:2px;margin-right:6px}
.chip.green{color:#4dff88;border-color:#1a4030;background:#0d1a20}
.chip.red{color:#ff6060;border-color:#401a1a;background:#1a0d0d}
.chip.warn{color:#ffcc44;border-color:#403010;background:#1a1400}
.det-row{background:#0d1520;border-left:3px solid #4d9fff;border-radius:4px;padding:8px 12px;margin-bottom:6px;font-size:.78rem}
.det-row span{color:#4d9fff;font-family:'Syne',sans-serif}
.det-row small{color:#2a4a7a;float:right}
div.stButton>button{background:#4d9fff;color:#080c14;border:none;border-radius:4px;font-family:'Syne',sans-serif;font-weight:700;letter-spacing:1px;padding:.5rem 1.2rem;transition:all .2s}
div.stButton>button:hover{background:#7dbfff;box-shadow:0 0 16px #4d9fff55}
#MainMenu,footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
ANIMAL_CLASSES = {
    14:"bird",15:"cat",16:"dog",17:"horse",18:"sheep",
    19:"cow",20:"elephant",21:"bear",22:"zebra",23:"giraffe",
}
DATA_DIR = Path("collected_data")
DATA_DIR.mkdir(exist_ok=True)

# ── Session state ──────────────────────────────────────────────────────────────
def _init():
    defs = {
        "cameras":[
            {"name":"Phone 1","url":"http://192.168.1.101:8080/shot.jpg","enabled":True},
            {"name":"Phone 2","url":"http://192.168.1.102:8080/shot.jpg","enabled":False},
        ],
        "active_cam":0,"running":False,"collect":False,
        "collect_count":0,"frame_count":0,"last_dets":[],
    }
    for k,v in defs.items():
        if k not in st.session_state:
            st.session_state[k]=v
_init()

# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading YOLO26 model…")
def load_model(name):
    from ultralytics import YOLO
    return YOLO(name)

# ── Frame fetch (PIL, no cv2 required) ────────────────────────────────────────
def fetch_frame(url: str, timeout: int = 3):
    """Fetch one PIL Image from a phone camera URL."""
    try:
        if url.endswith((".jpg",".jpeg","/shot.jpg","/jpeg")):
            # Snapshot mode — pure Python, works everywhere
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return Image.open(BytesIO(r.read())).convert("RGB")
        elif CV2_AVAILABLE:
            # Stream mode (MJPEG/RTSP) — needs cv2
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception:
        pass
    return None

# ── Draw boxes on PIL (no cv2) ────────────────────────────────────────────────
def draw_boxes(img: Image.Image, dets: list, show_lbl: bool, show_conf: bool) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for d in dets:
        x1,y1,x2,y2 = d["bbox"]
        draw.rectangle([x1,y1,x2,y2], outline=(77,159,255), width=2)
        if show_lbl or show_conf:
            parts=[]
            if show_lbl:  parts.append(d["label"].upper())
            if show_conf: parts.append(f"{d['conf']:.0%}")
            txt=" ".join(parts)
            tw = int(draw.textlength(txt)) + 8
            draw.rectangle([x1,y1-18,x1+tw,y1], fill=(20,60,120))
            draw.text((x1+4,y1-16), txt, fill=(180,220,255))
    return img

# ── Threaded camera reader ─────────────────────────────────────────────────────
class CameraReader:
    def __init__(self, url: str):
        self.url=url; self.frame=None; self.ok=False
        self._lock=threading.Lock(); self._stop=threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        while not self._stop.is_set():
            img=fetch_frame(self.url)
            with self._lock:
                self.ok = img is not None
                if img: self.frame=img
            time.sleep(0.04)
    def read(self):
        with self._lock:
            return self.ok, self.frame.copy() if self.frame else None
    def stop(self): self._stop.set()

# ── YOLO detection ─────────────────────────────────────────────────────────────
def run_detect(model, pil_img, conf, iou, animals_only, show_lbl, show_conf, track):
    arr = np.array(pil_img)
    kw = dict(conf=conf, iou=iou, verbose=False)
    results = model.track(arr, persist=True, **kw) if track else model.predict(arr, **kw)
    result = results[0]; dets=[]
    if result.boxes is not None:
        for box in result.boxes:
            cls_id=int(box.cls[0]); conf_v=float(box.conf[0])
            if animals_only and cls_id not in ANIMAL_CLASSES: continue
            label=ANIMAL_CLASSES.get(cls_id, result.names.get(cls_id,str(cls_id)))
            x1,y1,x2,y2=map(int,box.xyxy[0])
            dets.append({"label":label,"conf":conf_v,"bbox":(x1,y1,x2,y2)})
    annotated=draw_boxes(pil_img, dets, show_lbl, show_conf)
    return annotated, dets

# ══ SIDEBAR ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    if not CV2_AVAILABLE:
        st.warning("⚠️ cv2 not available — snapshot mode only. Use `/shot.jpg` URLs.")
    model_name  = st.selectbox("Model",["yolo26n.pt","yolo26s.pt","yolo26m.pt","yolo26l.pt"])
    conf_thr    = st.slider("Confidence",0.10,0.95,0.40,0.05)
    iou_thr     = st.slider("IoU (NMS)", 0.10,0.95,0.45,0.05)
    st.markdown("---")
    animals_only= st.toggle("🐾 Animals Only",   True)
    show_lbl    = st.toggle("Show Labels",        True)
    show_conf_v = st.toggle("Show Confidence",    True)
    enable_track= st.toggle("Enable Tracking",    False)
    st.markdown("---")
    st.markdown("### 📷 Camera Setup")
    st.caption("IP Webcam app (Android) → Start Server → use `/shot.jpg` URL")
    for i,cam in enumerate(st.session_state.cameras):
        with st.expander(f"{cam['name']} {'🟢' if cam['enabled'] else '⚫'}"):
            cam["name"]   =st.text_input("Name",      cam["name"],   key=f"cn_{i}")
            cam["url"]    =st.text_input("URL",        cam["url"],    key=f"cu_{i}")
            cam["enabled"]=st.checkbox("Enabled",      cam["enabled"],key=f"ce_{i}")
    st.markdown("---")
    st.markdown("### 💾 Data Collection")
    save_dir =st.text_input("Save folder",str(DATA_DIR))
    save_raw =st.checkbox("Save raw frames",      True)
    save_ann =st.checkbox("Save annotated frames",True)
    save_json=st.checkbox("Save detection JSON",  True)
    every_n  =st.number_input("Every N frames",1,60,5)
    if st.button("🗑 Clear data"):
        import shutil; shutil.rmtree(save_dir,ignore_errors=True)
        Path(save_dir).mkdir(exist_ok=True)
        st.session_state.collect_count=0; st.success("Cleared.")
    st.markdown("---")
    cv2_stat='✅' if CV2_AVAILABLE else '❌ snapshot-only'
    st.markdown(f'<div style="font-size:.65rem;color:#1a3060">YOLO26 · CV2:{cv2_stat}</div>',unsafe_allow_html=True)

# ══ HEADER ═════════════════════════════════════════════════════════════════════
c1,c2=st.columns([3,1])
with c1:
    st.markdown('<p class="hero">MULTI-CAM<br>ANIMAL DETECTOR</p>',unsafe_allow_html=True)
    st.markdown('<p class="sub">2-Phone Live Feed · YOLO26 · Data Collection</p>',unsafe_allow_html=True)
with c2:
    n=len([c for c in st.session_state.cameras if c["enabled"]])
    cv_chip='<span class="chip green">CV2 ✅</span>' if CV2_AVAILABLE else '<span class="chip warn">SNAPSHOT</span>'
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown(f'<span class="chip">{n} CAMS</span>{cv_chip}',unsafe_allow_html=True)
st.markdown("---")

# ── Camera switcher ────────────────────────────────────────────────────────────
st.markdown("#### 🔄 Active Camera")
cam_cols=st.columns(2)
for i,(col,cam) in enumerate(zip(cam_cols,st.session_state.cameras)):
    with col:
        active=st.session_state.active_cam==i
        border="border:2px solid #4d9fff;" if active else "border:1px solid #1a2a4a;"
        st.markdown(
            f'<div class="cam-card {"active" if active else ""}" style="{border}">'
            f'<div class="cam-label">{cam["name"]}</div>'
            f'<div class="cam-url">{cam["url"]}</div>'
            f'<div class="{"status-ok" if cam["enabled"] else "status-bad"}">'
            f'{"🟢 ON" if cam["enabled"] else "⚫ OFF"}</div></div>',
            unsafe_allow_html=True)
        if st.button("Switch →",key=f"sw_{i}",disabled=not cam["enabled"]):
            st.session_state.active_cam=i; st.rerun()
st.markdown("---")

# ── Controls ───────────────────────────────────────────────────────────────────
c1,c2,c3,c4=st.columns(4)
with c1:
    if st.button("▶ Start Detection"):   st.session_state.running=True
with c2:
    if st.button("⏹ Stop"):              st.session_state.running=False
with c3:
    if st.button("💾 Start Collecting"): st.session_state.collect=True; Path(save_dir).mkdir(parents=True,exist_ok=True)
with c4:
    if st.button("⏸ Stop Collecting"):  st.session_state.collect=False

stat_ph=st.empty()
mc,sc=st.columns([3,1])
with mc:
    st.markdown("**Live Feed**")
    orig_ph=st.empty(); annot_ph=st.empty()
with sc:
    st.markdown("**Detections**")
    det_ph=st.empty()

st.markdown("---"); st.markdown("**All Camera Feeds**")
th_cols=st.columns(2); th_phs=[c.empty() for c in th_cols]
for c,cam in zip(th_cols,st.session_state.cameras): c.caption(cam["name"])

# ══ MAIN LOOP ══════════════════════════════════════════════════════════════════
if st.session_state.running:
    active_idx=st.session_state.active_cam
    active_cam=st.session_state.cameras[active_idx]
    if not active_cam["enabled"]:
        st.error(f"{active_cam['name']} is disabled."); st.stop()

    url=active_cam["url"]
    is_snap=url.endswith((".jpg",".jpeg","/shot.jpg","/jpeg"))
    if not CV2_AVAILABLE and not is_snap:
        st.error(
            "**cv2 not available** on this server.\n\n"
            f"Change URL to snapshot mode: `{url.rsplit('/',1)[0]}/shot.jpg`"
        ); st.stop()

    model=load_model(model_name)
    readers={i:CameraReader(cam["url"]) for i,cam in enumerate(st.session_state.cameras) if cam["enabled"]}
    time.sleep(1.2)

    frame_idx=0; save_path=Path(save_dir)
    while st.session_state.running:
        reader=readers.get(active_idx)
        if not reader: break
        ok,pil_img=reader.read()
        if not ok or pil_img is None:
            stat_ph.markdown(f'<span class="chip red">❌ {active_cam["name"]} — no frame</span>',unsafe_allow_html=True)
            time.sleep(0.25); continue

        t0=time.perf_counter()
        annotated,dets=run_detect(model,pil_img,conf_thr,iou_thr,animals_only,show_lbl,show_conf_v,enable_track)
        elapsed=(time.perf_counter()-t0)*1000

        orig_ph.image(pil_img,use_container_width=True)
        annot_ph.image(annotated,use_container_width=True)

        ctag=f'<span class="chip green">💾 {st.session_state.collect_count} saved</span>' if st.session_state.collect else '<span class="chip">💾 off</span>'
        stat_ph.markdown(
            f'<span class="chip">📷 {active_cam["name"]}</span>'
            f'<span class="chip">⚡ {elapsed:.0f} ms</span>'
            f'<span class="chip green">🐾 {len(dets)} animals</span>'
            f'<span class="chip">Frame {frame_idx}</span>{ctag}',
            unsafe_allow_html=True)

        det_html="".join(
            f'<div class="det-row"><span>🐾 {d["label"].upper()}</span><small>{d["conf"]:.0%}</small>'
            f'<br><span style="color:#1a3060;font-size:.65rem">{"█"*int(d["conf"]*8)}</span></div>'
            for d in sorted(dets,key=lambda x:-x["conf"])
        ) if dets else '<div style="color:#2a4a7a;font-size:.75rem">No animals detected</div>'
        det_ph.markdown(det_html,unsafe_allow_html=True)

        for i,(cam,t_ph) in enumerate(zip(st.session_state.cameras,th_phs)):
            if not cam["enabled"]:
                t_ph.markdown('<div style="background:#0d1520;border:1px solid #1a2a4a;height:80px;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#1a3060;font-size:.65rem">DISABLED</div>',unsafe_allow_html=True)
                continue
            r=readers.get(i)
            if r:
                ok_t,f_t=r.read()
                if ok_t and f_t: t_ph.image(f_t,use_container_width=True)

        if st.session_state.collect and frame_idx%every_n==0:
            ts=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            pfx=save_path/f"{active_cam['name'].replace(' ','_')}_{ts}"
            if save_raw: pil_img.save(str(pfx)+"_raw.jpg")
            if save_ann: annotated.save(str(pfx)+"_ann.jpg")
            if save_json and dets:
                with open(str(pfx)+"_dets.json","w") as jf:
                    json.dump({"camera":active_cam["name"],"url":active_cam["url"],"timestamp":ts,"model":model_name,
                        "detections":[{"label":d["label"],"confidence":round(d["conf"],4),"bbox":list(d["bbox"])} for d in dets]},jf,indent=2)
            st.session_state.collect_count+=1

        frame_idx+=1; st.session_state.frame_count=frame_idx
        time.sleep(0.03)

    for r in readers.values(): r.stop()

else:
    for ph in [orig_ph,annot_ph]:
        ph.markdown('<div style="background:#0d1520;border:1px dashed #1a2a4a;border-radius:8px;height:200px;display:flex;align-items:center;justify-content:center;color:#1a3060;font-size:.85rem;letter-spacing:3px">PRESS ▶ START</div>',unsafe_allow_html=True)
    if st.session_state.collect_count>0:
        st.markdown(f'<span class="chip green">💾 {st.session_state.collect_count} frames in {save_dir}</span>',unsafe_allow_html=True)

st.markdown("---")
with st.expander("📁 Collected Data Preview",expanded=False):
    imgs=sorted(Path(save_dir).glob("*_ann.jpg"))[-8:]
    if imgs:
        cols=st.columns(min(4,len(imgs)))
        for col,p in zip(cols*2,imgs): col.image(str(p),caption=p.stem[-20:],use_container_width=True)
    else:
        st.info("No collected frames yet.")
    jsons=sorted(Path(save_dir).glob("*_dets.json"))
    if jsons:
        st.markdown(f"**{len(jsons)} JSON files**")
        with open(jsons[-1]) as f: st.json(json.load(f))

st.markdown("---")
st.markdown('<div style="text-align:center;font-size:.65rem;color:#0d1a30;letter-spacing:3px">MULTI-CAM ANIMALS DETECTION · YOLO26 · ULTRALYTICS · STREAMLIT</div>',unsafe_allow_html=True)
