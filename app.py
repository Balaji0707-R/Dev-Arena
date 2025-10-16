"""
Stylish PPE Compliance Tracker
Features:
- Modern UI with CSS
- `st.camera_input` for browser webcam capture
- Local YOLO .pt model upload (or fallback to yolov8n)
- Image and video upload processing
"""
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import io
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="PPE Compliance Tracker", layout="wide")

_CSS = '''
<style>
body { background: linear-gradient(135deg,#0f2027,#203a43,#2c5364); }
.card { background: rgba(255,255,255,0.04); border-radius: 12px; padding: 16px; }
.title { font-size:28px; font-weight:700; color:#e6eef3 }
.muted { color:#bcd6e6 }
.controls { background: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px }
</style>
'''

st.markdown(_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_yolo_model(path: str = None):
    """Load a YOLO model. If path is None, load yolov8n."""
    try:
        if path and os.path.exists(path):
            return YOLO(path)
        return YOLO('yolov8n')
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def predict_frame(model, frame: np.ndarray):
    try:
        res = model(frame)[0]
    except Exception as e:
        st.warning(f"Inference error: {e}")
        return frame
    boxes = getattr(res, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        return frame
    try:
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
    except Exception:
        return frame

    for (x1, y1, x2, y2), c, conf in zip(xyxy, cls, confs):
        color = (0,255,0) if int(c) == 0 else (255,0,0)
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
        cv2.putText(frame, f"{int(c)} {conf:.2f}", (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def process_video_file(path: str, model):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        st.error('Unable to open video file')
        return
    stframe = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = predict_frame(model, frame)
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        stframe.image(rgb, channels='RGB', use_column_width=True)
    cap.release()


def process_image_bytes(image_bytes: bytes, model):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    out = predict_frame(model, frame)
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    st.image(rgb, use_column_width=True)


def main():
    st.markdown('<div class="title">PPE Compliance Tracker</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Upload a YOLO .pt model or use the default yolov8n</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_model = st.file_uploader('Upload YOLO .pt model (optional)', type=['pt'])
        if uploaded_model is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
            tmp.write(uploaded_model.getbuffer())
            tmp.flush()
            tmp.close()
            model = load_yolo_model(tmp.name)
        else:
            model = load_yolo_model(None)
        st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])
    with col2:
        st.markdown('<div class="card controls">', unsafe_allow_html=True)
        st.write('Input')
        option = st.radio('Select input', ('Camera (browser)','Upload'))
        st.markdown('</div>', unsafe_allow_html=True)

    with col1:
        if option == 'Camera (browser)':
            cam = st.camera_input('Use your browser camera')
            if cam is not None:
                img_bytes = cam.getvalue()
                process_image_bytes(img_bytes, model)
        else:
            uploaded = st.file_uploader('Upload image or video', type=['jpg','png','mp4'])
            if uploaded is not None:
                ext = uploaded.name.split('.')[-1].lower()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.'+ext)
                tmp.write(uploaded.getbuffer())
                tmp.flush()
                tmp.close()
                if ext in ('jpg','png'):
                    with open(tmp.name,'rb') as f:
                        data = f.read()
                    process_image_bytes(data, model)
                else:
                    process_video_file(tmp.name, model)
                try:
                    os.remove(tmp.name)
                except Exception:
                    pass


if __name__ == '__main__':
    main()
"""
PPE Compliance Tracker
Run with: streamlit run app.py

This app uses the Roboflow hosted detect API to run PPE detection (helmet/vest/no-helmet/no-vest).
Place your Roboflow API key and model id in the sidebar. The app supports Live Webcam and Upload (image/video).
"""
import streamlit as st
import cv2
import numpy as np
import requests
import time
import tempfile
import os
import io
from PIL import Image

try:
    from supervision import BoxAnnotator, Detections
except Exception:
    BoxAnnotator = None
    Detections = None


st.set_page_config(page_title="PPE Compliance Tracker", layout="wide")


_CSS = '''
<style>
.stApp { background: linear-gradient(120deg, #0f2027, #203a43, #2c5364); color: #e6eef3 }
.card { background: rgba(255,255,255,0.03); padding: 18px; border-radius: 12px }
.title { font-size: 28px; font-weight: 700 }
</style>
'''

st.markdown(_CSS, unsafe_allow_html=True)


def build_roboflow_url(model_id: str, api_key: str) -> str:
    model_id = model_id.strip().strip('/')
    return f"https://detect.roboflow.com/{model_id}?api_key={api_key}"


def roboflow_predict_image(frame: np.ndarray, model_url: str, confidence: float = 0.25, timeout: int = 30):
    ok, buf = cv2.imencode('.jpg', frame)
    if not ok:
        return []
    files = {"file": ('image.jpg', buf.tobytes(), 'image/jpeg')}
    params = {"confidence": confidence}
    try:
        r = requests.post(model_url, params=params, files=files, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.warning(f"Roboflow request failed: {e}")
        return []

    preds = []
    for p in data.get('predictions', []):
        x = p.get('x')
        y = p.get('y')
        w = p.get('width')
        h = p.get('height')
        label = p.get('class') or p.get('label') or p.get('name') or 'object'
        conf = p.get('confidence', 0)
        if None in (x, y, w, h):
            xmin = p.get('x_min') or p.get('xmin')
            ymin = p.get('y_min') or p.get('ymin')
            xmax = p.get('x_max') or p.get('xmax')
            ymax = p.get('y_max') or p.get('ymax')
            if None not in (xmin, ymin, xmax, ymax):
                preds.append({'x1': int(xmin), 'y1': int(ymin), 'x2': int(xmax), 'y2': int(ymax), 'class': label, 'confidence': conf})
            continue
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        preds.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': label, 'confidence': conf})
    return preds


def draw_predictions(frame: np.ndarray, preds: list):
    if BoxAnnotator is None or Detections is None:
        for p in preds:
            cls = p.get('class','')
            conf = p.get('confidence',0)
            color = (0,255,0) if 'no' not in cls.lower() else (0,0,255)
            x1,y1,x2,y2 = p['x1'],p['y1'],p['x2'],p['y2']
            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            cv2.putText(frame, f"{cls} {conf:.2f}", (x1, max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    boxes = []
    scores = []
    class_ids = []
    labels = []
    name_to_id = {}
    next_id = 0
    for p in preds:
        boxes.append([p['x1'], p['y1'], p['x2'], p['y2']])
        scores.append(float(p.get('confidence',0)))
        name = p.get('class','')
        if name not in name_to_id:
            name_to_id[name] = next_id
            next_id += 1
        class_ids.append(name_to_id[name])
        labels.append(f"{name} {p.get('confidence',0):.2f}")

    detections = Detections.from_xyxy(np.array(boxes), scores=np.array(scores), class_id=np.array(class_ids))
    box_annotator = BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    return frame


def process_video_file(path: str, model_url: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        st.error('Unable to open video file')
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stframe = st.empty()
    progress = st.progress(0)
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        preds = roboflow_predict_image(frame, model_url)
        out = draw_predictions(frame, preds)
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        stframe.image(rgb, channels='RGB', use_column_width=True)
        if total:
            progress.progress(min(frame_no/total,1.0))
    cap.release()
    progress.empty()


def process_image_bytes(image_bytes: bytes, model_url: str):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    preds = roboflow_predict_image(frame, model_url)
    out = draw_predictions(frame, preds)
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    st.image(rgb, use_column_width=True)


def main():
    st.title('PPE Compliance Tracker')
    st.markdown('<div class="subtitle">Detect helmets and vests using your Roboflow model</div>', unsafe_allow_html=True)

    with st.sidebar.expander('How to use / Roboflow setup', expanded=True):
        st.write('Insert your Roboflow API key and model id (example: my-ppe-model/1) below.')
        st.write('Obtain an inference URL from Roboflow or use API key + model id.')
        st.markdown('---')
        st.write('Note: this app uses the Roboflow Hosted Detect endpoint. Ensure your model is deployed for inference.')

    api_key = st.sidebar.text_input('Roboflow API Key', value='')
    model_id = st.sidebar.text_input('Roboflow Model ID (e.g. ppe-model/1)', value='')
    model_url = None
    if api_key and model_id:
        model_url = build_roboflow_url(model_id, api_key)

    mode = st.radio('Mode', ('Live Webcam Detection','Upload Image/Video'))

    if mode == 'Live Webcam Detection':
        col1, col2 = st.columns([3,1])
        with col2:
            st.write('Webcam Controls')
            start = st.button('Start Webcam')
            stop = st.button('Stop Webcam')
        if start:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error('Unable to open webcam')
            else:
                stframe = col1.empty()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    preds = roboflow_predict_image(frame, model_url) if model_url else []
                    out = draw_predictions(frame, preds)
                    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                    stframe.image(rgb, channels='RGB', use_column_width=True)
                    # break if stop pressed
                    if stop:
                        break
                cap.release()

    else:
        uploaded = st.file_uploader('Upload image (.jpg/.png) or video (.mp4)', type=['jpg','png','mp4'])
        if uploaded is not None:
            if not (api_key and model_id):
                st.warning('Please enter Roboflow API key and model id in the sidebar')
            else:
                ext = uploaded.name.split('.')[-1].lower()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.'+ext)
                tmp.write(uploaded.getbuffer())
                tmp.flush()
                tmp.close()
                if ext in ('jpg','png'):
                    with open(tmp.name,'rb') as f:
                        data = f.read()
                    process_image_bytes(data, model_url)
                else:
                    process_video_file(tmp.name, model_url)
                try:
                    os.remove(tmp.name)
                except Exception:
                    pass


if __name__ == '__main__':
    main()
"""
PPE Compliance Tracker
Run with: streamlit run app.py

Notes:
- Insert your Roboflow API key and model id in the sidebar (or environment variables).
- The app supports Live Webcam Detection and Upload Image/Video.
"""
import streamlit as st
import cv2
import numpy as np
import requests
import time
import tempfile
import os
from PIL import Image

try:
    from supervision import BoxAnnotator, Detections
except Exception:
    BoxAnnotator = None
    Detections = None

st.set_page_config(page_title="PPE Compliance Tracker", layout="wide")

_CSS = '''
<style>
.stApp { background: linear-gradient(120deg, #0f2027, #203a43, #2c5364); color: #e6eef3 }
.container { background: rgba(255,255,255,0.03); padding: 18px; border-radius: 12px }
.title { font-size: 28px; font-weight: 700 }
</style>
'''

st.markdown(_CSS, unsafe_allow_html=True)


def build_roboflow_url(model_id: str, api_key: str):
    model_id = model_id.strip().strip('/')
    return f"https://detect.roboflow.com/{model_id}?api_key={api_key}"


def roboflow_predict_image(image_bgr: np.ndarray, model_url: str, confidence: float = 0.25, timeout: int = 30):
    """Send image to Roboflow detect endpoint and return list of boxes."""
    ok, buf = cv2.imencode('.jpg', image_bgr)
    if not ok:
        return []
    files = {"file": ('image.jpg', buf.tobytes(), 'image/jpeg')}
    params = {"confidence": confidence}
    try:
        r = requests.post(model_url, params=params, files=files, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.error(f"Roboflow request failed: {e}")
        return []

    preds = []
    for p in data.get('predictions', []):
        # Roboflow returns center x,y,width,height in pixels for hosted detect
        x = p.get('x')
        y = p.get('y')
        w = p.get('width')
        h = p.get('height')
        cls = p.get('class') or p.get('label') or 'object'
        conf = p.get('confidence', 0)
        if None in (x, y, w, h):
            # fallback to xmin/xmax
            xmin = p.get('x_min') or p.get('xmin')
            ymin = p.get('y_min') or p.get('ymin')
            xmax = p.get('x_max') or p.get('xmax')
            ymax = p.get('y_max') or p.get('ymax')
            if None not in (xmin, ymin, xmax, ymax):
                preds.append({'x1': int(xmin), 'y1': int(ymin), 'x2': int(xmax), 'y2': int(ymax), 'class': cls, 'confidence': conf})
            continue
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        preds.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': cls, 'confidence': conf})
    return preds


def draw_predictions(frame_bgr: np.ndarray, preds: list):
    """Draw boxes and labels. Prefer supervision if available."""
    if BoxAnnotator is None or Detections is None:
        # simple opencv drawing
        for p in preds:
            cls = p.get('class', '')
            conf = p.get('confidence', 0)
            color = (0,255,0) if 'no' not in cls.lower() else (0,0,255)
            x1,y1,x2,y2 = p['x1'], p['y1'], p['x2'], p['y2']
            cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame_bgr, f"{cls} {conf:.2f}", (x1, max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame_bgr

    # Build supervision structures
    boxes = []
    scores = []
    class_ids = []
    labels = []
    name_to_id = {}
    next_id = 0
    for p in preds:
        boxes.append([p['x1'], p['y1'], p['x2'], p['y2']])
        scores.append(float(p.get('confidence', 0)))
        name = p.get('class','')
        if name not in name_to_id:
            name_to_id[name] = next_id
            next_id += 1
        class_ids.append(name_to_id[name])
        labels.append(f"{name} {p.get('confidence',0):.2f}")

    detections = Detections.from_xyxy(np.array(boxes), scores=np.array(scores), class_id=np.array(class_ids))
    box_annotator = BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    frame = box_annotator.annotate(scene=frame_bgr, detections=detections, labels=labels)
    return frame


def process_video_file(path: str, model_url: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        st.error('Unable to open video file')
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stframe = st.empty()
    progress = st.progress(0)
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        preds = roboflow_predict_image(frame, model_url)
        out = draw_predictions(frame, preds)
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        stframe.image(rgb, channels='RGB', use_column_width=True)
        if total:
            progress.progress(min(frame_no/total,1.0))
    cap.release()
    progress.empty()


def process_image_file(image_bytes: bytes, model_url: str):
    image = Image.open(tempfile.SpooledTemporaryFile())
    image = Image.open(tempfile.SpooledTemporaryFile())
    # safer: read via PIL from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    preds = roboflow_predict_image(frame, model_url)
    out = draw_predictions(frame, preds)
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    st.image(rgb, use_column_width=True)


def main():
    st.title('PPE Compliance Tracker')
    st.markdown('<div class="subtitle">Detect helmets and vests using your Roboflow model</div>', unsafe_allow_html=True)

    # Sidebar for Roboflow credentials and instructions
    with st.sidebar.expander('How to use / Roboflow setup', expanded=True):
        st.write('Insert your Roboflow API key and model id (example: my-ppe-model/1) below.')
        st.write('You can obtain an inference URL from Roboflow or use the API key + model id.')
        st.markdown('---')
        st.write('If you prefer, export your model to a .pt and use the ultralytics path-based loader in a different app variant.')

    api_key = st.sidebar.text_input('Roboflow API Key', value='')
    model_id = st.sidebar.text_input('Roboflow Model ID (e.g. ppe-model/1)', value='')
    model_url = None
    if api_key and model_id:
        model_url = build_roboflow_url(model_id, api_key)

    mode = st.radio('Mode', ('Live Webcam Detection','Upload Image/Video'))

    if mode == 'Live Webcam Detection':
        st.write('Starting webcam...')
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error('Unable to open webcam')
            return
        stframe = st.empty()
        stop_button = st.button('Stop')
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if model_url:
                preds = roboflow_predict_image(frame, model_url)
            else:
                preds = []
            out = draw_predictions(frame, preds)
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            stframe.image(rgb, channels='RGB', use_column_width=True)
            if stop_button:
                break
        cap.release()

    else:
        uploaded = st.file_uploader('Upload image (.jpg/.png) or video (.mp4)', type=['jpg','png','mp4'])
        if uploaded is not None:
            if model_url is None:
                st.warning('Please enter Roboflow API key and model id in the sidebar')
            else:
                ext = uploaded.name.split('.')[-1].lower()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.'+ext)
                tmp.write(uploaded.getbuffer())
                tmp.flush()
                tmp.close()
                if ext in ('jpg','png'):
                    with open(tmp.name,'rb') as f:
                        data = f.read()
                    process_image_file(data, model_url)
                else:
                    process_video_file(tmp.name, model_url)
                try:
                    os.remove(tmp.name)
                except Exception:
                    pass


if __name__ == '__main__':
    main()
import logging
import requests
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import time
from datetime import timedelta
from ultralytics import YOLO


st.set_page_config(page_title="Safe-Sight AI: Forbidden Zone Detector", layout="wide")


@st.cache_resource
def load_model(path: str = "yolov8n.pt", roboflow_url: str = None, uploaded_model_path: str = None):
    """Load and cache the YOLO model to avoid reloading between reruns.

    Behavior:
    - If `path` exists on disk, attempt to load it.
    - Otherwise, try to load the named model 'yolov8n' which triggers ultralytics to download the weights.
    """
    try:
        # If an uploaded model path was provided by the user, prefer it
        if uploaded_model_path and os.path.exists(uploaded_model_path):
            logging.info(f"Loading YOLO model from uploaded file: {uploaded_model_path}")
            return YOLO(uploaded_model_path)

        # If a Roboflow/remote URL was provided, download it to a temp file and load
        if roboflow_url:
            logging.info(f"Downloading model from URL: {roboflow_url}")
            try:
                r = requests.get(roboflow_url, stream=True, timeout=60)
                r.raise_for_status()
                fd, tmp_path = tempfile.mkstemp(suffix=".pt")
                os.close(fd)
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                logging.info(f"Downloaded model to: {tmp_path}")
                return YOLO(tmp_path)
            except Exception as e:
                logging.exception("Failed to download model from provided URL")
                raise RuntimeError(f"Failed to download model from provided URL: {e}")

        # Prefer local file if present
        if path and os.path.exists(path):
            logging.info(f"Loading YOLO model from local file: {path}")
            return YOLO(path)

        # If local file missing, attempt to load by model name so ultralytics will download it
        logging.info("Local weights not found; attempting to download 'yolov8n' automatically.")
        st.info("YOLOv8 weights not found locally — downloading weights (this may take a few minutes)...")
        return YOLO("yolov8n")
    except Exception as e:
        logging.exception("Failed to load YOLO model")
        # Re-raise so the caller can present an error
        raise RuntimeError(f"Failed to load YOLO model: {e}")


def format_timestamp(ms: float) -> str:
    """Convert milliseconds to H:M:S.ms formatted string."""
    try:
        td = timedelta(milliseconds=int(ms))
        return str(td)
    except Exception:
        return "0:00:00"


def process_video(video_path: str, x_start: int, x_end: int, alerts: list, model: YOLO):
    """
    Process video frame-by-frame, run YOLOv8 inference, annotate frames, and collect alerts.

    Yields annotated BGR frames for display. Alerts are appended to the provided `alerts` list
    as dictionaries: {'frame': int, 'timestamp': str, 'x_center': int, 'confidence': float}.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Unable to open the uploaded video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_no = 0

    # Normalize zone coordinates for safety (will be clipped to frame width per-frame)
    try:
        x_start = int(x_start)
        x_end = int(x_end)
    except Exception:
        x_start, x_end = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        h, w = frame.shape[:2]

        # Clip inputs to frame boundaries
        xs = max(0, min(w - 1, x_start))
        xe = max(0, min(w - 1, x_end))
        if xe < xs:
            xs, xe = xe, xs

        breach_in_frame = False

        # Run detection on the BGR frame (ultralytics accepts numpy arrays)
        try:
            results = model(frame)[0]
        except Exception as e:
            # If model fails on a frame, skip and continue
            st.warning(f"Model inference error on frame {frame_no}: {e}")
            yield frame
            continue

        boxes = getattr(results, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            # Extract arrays: xyxy, class ids, confidences
            try:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
            except Exception:
                # Fallback if the Boxes object API differs
                xyxy, cls, confs = [], [], []

            for (x1, y1, x2, y2), c, conf in zip(xyxy, cls, confs):
                # COCO class 0 corresponds to 'person'
                if int(c) != 0:
                    continue

                x_center = (float(x1) + float(x2)) / 2.0

                # Check if center falls inside forbidden zone
                if xs <= x_center <= xe:
                    breach_in_frame = True
                    # RED bounding box for breach
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    # Log the alert with timestamp from the video capture
                    ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    ts = format_timestamp(ms)
                    alerts.append({
                        "frame": frame_no,
                        "timestamp": ts,
                        "x_center": int(x_center),
                        "confidence": float(round(float(conf), 3)),
                    })
                else:
                    # GREEN bounding box for safe person
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Draw the forbidden zone overlay (red translucent rectangle)
        overlay = frame.copy()
        cv2.rectangle(overlay, (xs, 0), (xe, h), (0, 0, 255), -1)
        alpha = 0.12
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Add small HUD: frame number and breach status
        status_text = f"Frame: {frame_no} | Breach: {'YES' if breach_in_frame else 'NO'}"
        cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        yield frame

    cap.release()


def find_ppe_class_ids(model: YOLO):
    """Return lists of class IDs that likely correspond to helmets and safety jackets/vests.

    We perform a fuzzy match on the model's `names` mapping so the app works with custom PPE models
    that may use labels like 'helmet', 'hardhat', 'safety_vest', 'vest', 'jacket', etc.
    """
    helmet_keywords = ("helmet", "hardhat", "hard hat", "hat")
    jacket_keywords = ("jacket", "vest", "safety_vest", "safety jacket", "safety_jacket", "highvis", "hi-vis", "hi vis")
    helmet_ids = []
    jacket_ids = []
    try:
        names = getattr(model, "names", None) or {}
        for cid, name in names.items():
            lname = str(name).lower()
            for k in helmet_keywords:
                if k in lname:
                    helmet_ids.append(int(cid))
                    break
            for k in jacket_keywords:
                if k in lname:
                    jacket_ids.append(int(cid))
                    break
    except Exception:
        pass
    return list(set(helmet_ids)), list(set(jacket_ids))


def process_camera(source: int, x_start: int, x_end: int, alerts: list, model: YOLO, mode: str):
    """
    Process frames from a camera source (device index) and yield annotated frames.

    mode: 'Forbidden Zone' or 'PPE Detection'
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("Unable to access the camera. Check permissions and that the device is available.")
        return

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)

    # prepare PPE class ids
    helmet_ids, jacket_ids = find_ppe_class_ids(model)

    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        xs = max(0, min(w - 1, int(x_start)))
        xe = max(0, min(w - 1, int(x_end)))
        if xe < xs:
            xs, xe = xe, xs

        breach_in_frame = False

        try:
            results = model(frame)[0]
        except Exception as e:
            st.warning(f"Model inference error on camera frame {frame_no}: {e}")
            yield frame
            continue

        boxes = getattr(results, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            try:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
            except Exception:
                xyxy, cls, confs = [], [], []

            # For PPE mode, we'll draw person boxes and helmet/jacket boxes in separate colors
            for (x1, y1, x2, y2), c, conf in zip(xyxy, cls, confs):
                c = int(c)
                if mode == "PPE Detection":
                    if c in helmet_ids:
                        # Blue box for helmet
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(frame, f"Helmet {conf:.2f}", (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                    elif c in jacket_ids:
                        # Yellow box for jacket/vest
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                        cv2.putText(frame, f"Jacket {conf:.2f}", (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                    elif c == 0:
                        # person
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                else:
                    # Forbidden Zone logic (same as video)
                    if int(c) != 0:
                        continue
                    x_center = (float(x1) + float(x2)) / 2.0
                    if xs <= x_center <= xe:
                        breach_in_frame = True
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                        ts = format_timestamp(ms)
                        alerts.append({
                            "frame": frame_no,
                            "timestamp": ts,
                            "x_center": int(x_center),
                            "confidence": float(round(float(conf), 3)),
                        })
                    else:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Draw forbidden zone overlay if mode is Forbidden Zone
        if mode == "Forbidden Zone":
            overlay = frame.copy()
            cv2.rectangle(overlay, (xs, 0), (xe, h), (0, 0, 255), -1)
            alpha = 0.12
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        status_text = f"Frame: {frame_no} | Mode: {mode} | Breach: {'YES' if breach_in_frame else 'NO'}"
        cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        yield frame

    cap.release()


def save_uploaded_file(uploaded, dest_folder=None) -> str:
    """Save a Streamlit UploadedFile to disk and return the path."""
    suffix = os.path.splitext(uploaded.name)[1]
    if dest_folder is None:
        dest_folder = tempfile.gettempdir()
    fd, path = tempfile.mkstemp(suffix=suffix, dir=dest_folder)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


def main():
    st.title("Safe-Sight AI: Forbidden Zone Detector")

    st.sidebar.header("Forbidden Zone Settings")
    # We will present useful, scaled controls after the user uploads a video (so sliders can match frame width).
    ZONE_X_START = None
    ZONE_X_END = None
    st.sidebar.markdown("\nUpload a video and the sidebar will show sliders scaled to the video's width for easy tuning of the forbidden zone.")

    uploaded_video = st.file_uploader("Upload a video (.mp4 or .avi)", type=["mp4", "avi"])

    if uploaded_video is not None:
        video_path = save_uploaded_file(uploaded_video)

        col1, col2 = st.columns([2, 1])
        video_placeholder = col1.empty()
        log_placeholder = col2.empty()

        # Load model (cached)
        with st.spinner("Loading YOLOv8 model... this may take a moment"):
            model = load_model()

        alerts = []

        # Stream frames to the UI
        progress = st.progress(0)
        total_frames = None
        try:
            # Attempt to get total frames for progress and frame width for scaled sliders
            cap_tmp = cv2.VideoCapture(video_path)
            if cap_tmp.isOpened():
                total_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                frame_width = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            else:
                frame_width = 0
            cap_tmp.release()
        except Exception:
            total_frames = None
            frame_width = 0

        # Create sliders scaled to the video's frame width for easier zone selection
        if frame_width and frame_width > 1:
            default_start = int(frame_width * 0.25)
            default_end = int(frame_width * 0.75)
            ZONE_X_START = st.sidebar.slider("ZONE_X_START (px)", 0, frame_width - 1, default_start, key='zone_start')
            ZONE_X_END = st.sidebar.slider("ZONE_X_END (px)", 0, frame_width - 1, default_end, key='zone_end')
        else:
            # Fallback numeric inputs if width not available
            ZONE_X_START = st.sidebar.number_input("ZONE_X_START (pixels)", min_value=0, max_value=10000, value=100, step=1)
            ZONE_X_END = st.sidebar.number_input("ZONE_X_END (pixels)", min_value=0, max_value=10000, value=300, step=1)

        frame_count = 0
        for frame in process_video(video_path, ZONE_X_START, ZONE_X_END, alerts, model):
            frame_count += 1
            # Convert BGR -> RGB for display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            video_placeholder.image(rgb, channels="RGB", use_column_width=True)

            # Update log view with the most recent alerts
            if len(alerts) > 0:
                df = pd.DataFrame(alerts)
                log_placeholder.dataframe(df.sort_values(by=["frame", "timestamp"], ascending=[True, True]))
            else:
                log_placeholder.info("No breaches detected yet.")

            # Update progress if we know total frames
            if total_frames and total_frames > 0:
                progress.progress(min(frame_count / max(1, total_frames), 1.0))

            # Let Streamlit breathe between frames
            time.sleep(0.001)

        progress.empty()

        # Finalized alerts table
        st.subheader("Breach Alert Log")
        if len(alerts) > 0:
            df_final = pd.DataFrame(alerts)
            st.dataframe(df_final.sort_values(by=["frame", "timestamp"]))
            csv = df_final.to_csv(index=False).encode("utf-8")
            st.download_button("Download alerts CSV", csv, file_name="alerts.csv", mime="text/csv")
        else:
            st.info("No breaches were detected in the uploaded video.")

        # Cleanup temporary file
        try:
            os.remove(video_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()

# -----------------------
# requirements.txt
# -----------------------
# The following are the recommended dependencies for this app. Save them into a separate
# `requirements.txt` when preparing your deployment environment.
#
# streamlit
# ultralytics
# torch
# opencv-python
# numpy
# pandas
