# 📱 Multi-Phone Camera Setup Guide
## Animals Detection — 4 Phone Live Feed

---

## Step 1 — Phone Setup (Android)

### Install IP Webcam
1. Download **IP Webcam** from Play Store (free, by Pavel Khlebovich)
2. Open app → scroll to bottom → tap **"Start server"**
3. Note the IP shown on screen, e.g.:  `http://192.168.1.101:8080`

### Stream URLs to use in the app
| Stream type  | URL pattern                        |
|--------------|------------------------------------|
| MJPEG video  | `http://<ip>:8080/video`           |
| JPEG snapshot| `http://<ip>:8080/shot.jpg`        |
| RTSP stream  | `rtsp://<ip>:8080/h264_pcm.sdp`    |

> **Recommended for this app:** MJPEG (`/video`) — lowest latency, no extra setup.

---

## Step 2 — Network Requirements

- ✅ All phones and your PC on the **same WiFi network**
- ✅ No VPN active
- ✅ Phone screen stays on (disable sleep in IP Webcam settings)
- ✅ Recommended: 5GHz WiFi for lower latency

### Find phone IP addresses
- Android: Settings → WiFi → tap your network → IP address
- IP Webcam shows it automatically when server starts

---

## Step 3 — App Configuration

Edit the default URLs in the sidebar:

```
Phone 1: http://192.168.1.101:8080/video
Phone 2: http://192.168.1.102:8080/video
Phone 3: http://192.168.1.103:8080/video
Phone 4: http://192.168.1.104:8080/video
```

Replace `192.168.1.10x` with the actual IP shown in IP Webcam.

---

## Step 4 — Run the App

```bash
pip install ultralytics streamlit opencv-python-headless Pillow torch torchvision
streamlit run multi_cam_animals.py
```

---

## App Features

| Feature | Detail |
|---|---|
| **4 live feeds** | Simultaneous threaded readers, thumbnail grid |
| **Camera switcher** | One-click to switch active detection camera |
| **Animals-only filter** | 10 COCO animal classes |
| **Data collection** | Save raw frames, annotated frames, detection JSON |
| **Collect every N frames** | Controls dataset density |
| **JSON metadata** | Camera name, URL, timestamp, bboxes, confidence |
| **Live preview** | Shows last 8 collected annotated frames |

---

## Collected Data Structure

```
collected_data/
├── Phone_1_20250220_143201_raw.jpg        ← original frame
├── Phone_1_20250220_143201_ann.jpg        ← annotated with boxes
├── Phone_1_20250220_143201_dets.json      ← detection metadata
├── Phone_2_20250220_143215_raw.jpg
...
```

### JSON example
```json
{
  "camera": "Phone 1",
  "url": "http://192.168.1.101:8080/video",
  "timestamp": "20250220_143201_123456",
  "model": "yolo26n.pt",
  "detections": [
    {
      "label": "cow",
      "confidence": 0.8731,
      "bbox": [120, 80, 440, 360]
    }
  ]
}
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Black/no feed | Check IP, ensure IP Webcam server is running, same WiFi |
| Lag | Use 5GHz WiFi, reduce resolution in IP Webcam settings |
| App crashes | Fix PyTorch first (CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`) |
| Only 1 phone works | Check all 4 IPs individually in browser first |

---

## iOS Alternative
- **DroidCam** (iOS + Android): URL = `http://<ip>:4747/video`
- **Camo**: Use RTSP URL shown in the Camo app

---

## For Real Farm Deployment
- Mount phones on posts/walls at field angles
- Use PoE USB chargers to keep phones charged
- Set IP Webcam to max resolution + H.264
- Switch to `yolo26m.pt` or `yolo26l.pt` for better accuracy
- Enable tracking to count and follow individual animals
