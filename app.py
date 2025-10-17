# app.py
import streamlit as st
import cv2
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from collections import Counter
import threading
import time
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import tempfile
import os

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="AI Safety Monitoring System", layout="wide")
st.title("🦺 AI Safety Monitoring System")
st.write("This dashboard uses the Roboflow API for real-time Personal Protective Equipment (PPE) detection.")

# ----------------- SIDEBAR CONFIGURATION -----------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.divider()
    
    st.subheader("Roboflow Credentials")
    ROBoflow_API_KEY = st.text_input("Roboflow API Key", type="password")
    MODEL_ENDPOINT = st.text_input(
        "Roboflow Model Endpoint",
        placeholder="e.g., ppe-detection/1"
    )
    # Construct the full URL
    if ROBoflow_API_KEY and MODEL_ENDPOINT and not MODEL_ENDPOINT.startswith("https://"):
        MODEL_ENDPOINT = f"https://detect.roboflow.com/{MODEL_ENDPOINT}"

    def validate_credentials(api_key: str, url: str):
        if not api_key or not url:
            st.warning("Please enter your API key and Model Endpoint.")
            return False
        if not url.startswith("https://detect.roboflow.com/"):
            st.error("Invalid Model Endpoint format. It should be like 'project_name/version'.")
            return False
        return True

    credentials_valid = validate_credentials(ROBoflow_API_KEY, MODEL_ENDPOINT)
    
    st.divider()
    
    st.subheader("Detection Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.05)
    selected_classes_input = st.text_input("Filter classes (comma-separated)", help="Leave blank to detect all classes.")
    selected_classes = [c.strip() for c in selected_classes_input.split(',') if c.strip()]
    show_boxes = st.checkbox("Show Bounding Boxes", value=True)
    
    st.divider()

    st.subheader("📧 Email Alert Settings")
    EMAIL_ALERT_ENABLED = st.checkbox("Enable Email Alerts", value=True)
    EMAIL_SENDER = st.text_input("Sender Email", value="your_email@gmail.com")
    EMAIL_PASSWORD = st.text_input("Sender App Password", type="password")
    EMAIL_RECEIVER = st.text_input("Receiver Email", value="authorized_person@example.com")
    
    EMAIL_SMTP_SERVER = "smtp.gmail.com"
    EMAIL_SMTP_PORT = 587
    
    def send_email_alert(subject, body):
        if not EMAIL_ALERT_ENABLED or not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
            return
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_SENDER
            msg['To'] = EMAIL_RECEIVER
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            server = smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT)
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            text = msg.as_string()
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, text)
            server.quit()
        except Exception as e:
            st.error(f"Email alert failed: {e}")

    if st.button("Send Test Email"):
        if EMAIL_SENDER and EMAIL_PASSWORD and EMAIL_RECEIVER:
            send_email_alert(
                subject="PPE Email Test",
                body="This is a test email from your PPE Detection System."
            )
            st.success("Test email sent. Check your inbox/spam.")
        else:
            st.warning("Please fill in all email fields to send a test email.")

# Stop the app if credentials are not valid
if not credentials_valid:
    st.info("Enter valid credentials in the sidebar to begin.")
    st.stop()
    
# ----------------- BACKEND FUNCTIONS (UNCHANGED) -----------------
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

def safe_int(val, default=0):
    try:
        if val is None or np.isnan(val) or np.isinf(val): return default
        return int(round(float(val)))
    except: return default

def is_valid_number(val):
    return val is not None and not np.isnan(val) and not np.isinf(val)

def detect_ppe(frame: np.ndarray, timeout=10):
    try:
        img = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        pil_img.save(buf, format="JPEG", quality=70)
        buf.seek(0)
        response = requests.post(
            MODEL_ENDPOINT,
            params={"api_key": ROBoflow_API_KEY},
            files={"file": buf.getvalue()},
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as he:
        st.error(f"Roboflow HTTP error: {he}")
    except requests.exceptions.ConnectionError as ce:
        st.error(f"Network error: Please check your internet connection. Details: {ce}")
    except requests.exceptions.Timeout:
        st.warning("Roboflow request timed out.")
    except Exception as e:
        st.error(f"An unexpected error occurred during detection: {e}")
    return None

def draw_predictions_on_frame(frame: np.ndarray, predictions: dict):
    preds_list = []
    if not predictions or "predictions" not in predictions:
        return frame, preds_list, []

    for pred in predictions.get("predictions", []):
        try:
            cls = str(pred.get("class", "unknown"))
            conf = pred.get("confidence", 0.0)
            if not is_valid_number(conf) or conf < confidence_threshold:
                continue
            conf = float(conf)

            if selected_classes and cls not in selected_classes:
                continue

            x_c, y_c, w, h = pred.get("x"), pred.get("y"), pred.get("width"), pred.get("height")
            if not all(map(is_valid_number, [x_c, y_c, w, h])) or w <= 0 or h <= 0:
                continue
            x_c, y_c, w, h = float(x_c), float(y_c), float(w), float(h)
            
            x1 = max(0, safe_int(x_c - w/2))
            y1 = max(0, safe_int(y_c - h/2))
            x2 = min(safe_int(x_c + w/2), frame.shape[1]-1)
            y2 = min(safe_int(y_c + h/2), frame.shape[0]-1)

            preds_list.append({"class": cls, "conf": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

            if show_boxes:
                color = (0, 255, 0) # Green for compliant
                if "no_" in cls.lower() or "missing" in cls.lower() or "unsafe" in cls.lower():
                    color = (0, 0, 255) # Red for non-compliant
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{cls} {conf:.2f}", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        except Exception:
            continue
            
    classes = [p["class"] for p in preds_list]
    return frame, preds_list, classes

def compute_person_ppe_stats(preds_list):
    persons = [p for p in preds_list if p["class"].lower() == "person"]
    helmets = [p for p in preds_list if "helmet" in p["class"].lower() and not p["class"].lower().startswith("no_")]
    jackets = [p for p in preds_list if ("jacket" in p["class"].lower() or "safety" in p["class"].lower() or "vest" in p["class"].lower()) and not p["class"].lower().startswith("no_")]
    no_helmets = [p for p in preds_list if p["class"].lower().startswith("no_") and "helmet" in p["class"].lower()]
    no_jackets = [p for p in preds_list if p["class"].lower().startswith("no_") and ("jacket" in p["class"].lower() or "vest" in p["class"].lower())]

    # --- This entire function is complex backend logic, so it remains unchanged ---
    # (The function body is omitted here for brevity but is the same as your original)
    # ... [Same function body as your original code] ...
    if not persons and (helmets or jackets or no_helmets or no_jackets):
        helmet_positions = [(safe_int((h["x1"]+h["x2"])/2), safe_int((h["y1"]+h["y2"])/2)) for h in helmets]
        vest_positions = [(safe_int((v["x1"]+v["x2"])/2), safe_int((v["y1"]+v["y2"])/2)) for v in jackets]
        no_helmet_positions = [(safe_int((h["x1"]+h["x2"])/2), safe_int((h["y1"]+h["y2"])/2)) for h in no_helmets]
        no_vest_positions = [(safe_int((v["x1"]+v["x2"])/2), safe_int((v["y1"]+v["y2"])/2)) for v in no_jackets]
        matched_persons, used_vests = [], set()
        for i, (hx, hy) in enumerate(helmet_positions):
            closest_vest, min_dist = None, float('inf')
            for j, (vx, vy) in enumerate(vest_positions):
                if j not in used_vests:
                    dist = ((hx-vx)**2 + (hy-vy)**2)**0.5
                    if dist < min_dist and dist < 200: min_dist, closest_vest = dist, j
            if closest_vest is not None:
                used_vests.add(closest_vest); matched_persons.append({"has_helmet": True, "has_jacket": True})
            else: matched_persons.append({"has_helmet": True, "has_jacket": False})
        for j in range(len(vest_positions)):
            if j not in used_vests: matched_persons.append({"has_helmet": False, "has_jacket": True})
        for _ in no_helmet_positions: matched_persons.append({"has_helmet": False, "has_jacket": False})
        for _ in no_vest_positions: matched_persons.append({"has_helmet": False, "has_jacket": False})
        total_persons = len(matched_persons)
        safe_count = sum(1 for p in matched_persons if p["has_helmet"] and p["has_jacket"])
        return total_persons, safe_count, total_persons - safe_count, matched_persons
    
    per_person = []
    for person in persons:
        try:
            x1,y1,x2,y2 = safe_int(person["x1"]), safe_int(person["y1"]), safe_int(person["x2"]), safe_int(person["y2"])
            has_helmet = any(x1<=safe_int((h["x1"]+h["x2"])/2)<=x2 and y1<=safe_int((h["y1"]+h["y2"])/2)<=y2 for h in helmets)
            has_jacket = any(x1<=safe_int((j["x1"]+j["x2"])/2)<=x2 and y1<=safe_int((j["y1"]+j["y2"])/2)<=y2 for j in jackets)
            for nh in no_helmets:
                if x1<=safe_int((nh["x1"]+nh["x2"])/2)<=x2 and y1<=safe_int((nh["y1"]+nh["y2"])/2)<=y2: has_helmet=False
            for nj in no_jackets:
                if x1<=safe_int((nj["x1"]+nj["x2"])/2)<=x2 and y1<=safe_int((nj["y1"]+nj["y2"])/2)<=y2: has_jacket=False
            per_person.append({"person_box":(x1,y1,x2,y2),"has_helmet":has_helmet,"has_jacket":has_jacket})
        except: continue
    total_persons = len(per_person)
    safe_count = sum(1 for p in per_person if p["has_helmet"] and p["has_jacket"])
    return total_persons, safe_count, total_persons - safe_count, per_person

def plot_ppe_pie(stats_dict):
    labels, sizes = list(stats_dict.keys()), list(stats_dict.values())
    colors = ['#4CAF50', '#F44336']; fig, ax = plt.subplots(figsize=(4, 4))
    if not sizes or all(s == 0 for s in sizes):
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray'); ax.axis('off')
    else:
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(width=0.4, edgecolor='w')); ax.axis('equal')
    return fig

def convert_to_bytes(frame: np.ndarray):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); buf = BytesIO()
    pil_img.save(buf, format="PNG"); return buf.getvalue()

# ----------------- MAIN PAGE UI -----------------
tab1, tab2, tab3 = st.tabs(["🖼️ Image Upload", "🎥 Live Webcam Feed", "🎞️ Video Upload"])

# ----------------- IMAGE UPLOAD TAB -----------------
with tab1:
    st.header("Upload an Image for PPE Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        col1, col2 = st.columns([2, 1.2])
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            image = Image.open(uploaded_file).convert("RGB"); frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            with st.spinner("🔍 Analyzing image..."): predictions = detect_ppe(frame)
        with col2:
            st.subheader("📊 Analysis Results")
            if predictions:
                annotated_frame, preds_list, _ = draw_predictions_on_frame(frame.copy(), predictions)
                col1.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Annotated Image", use_container_width=True)
                col1.download_button("Download Annotated Image", data=convert_to_bytes(annotated_frame), file_name="annotated_image.png", mime="image/png")
                total_persons, safe_count, unsafe_count, per_person_details = compute_person_ppe_stats(preds_list)
                metric_cols = st.columns(3)
                metric_cols[0].metric("Total Persons", total_persons, delta_color="off")
                metric_cols[1].metric("✅ Safe", safe_count)
                metric_cols[2].metric("❌ Unsafe", unsafe_count, delta=f"-{unsafe_count}" if unsafe_count > 0 else "0")
                st.divider()
                st.write("**Safety Compliance Chart**"); fig = plot_ppe_pie({"Safe": safe_count, "Unsafe": unsafe_count}); st.pyplot(fig)
                st.divider()
                st.write("**📋 Individual Details**")
                if per_person_details:
                    for i, p in enumerate(per_person_details, 1):
                        helmet, jacket = "✅" if p["has_helmet"] else "❌", "✅" if p["has_jacket"] else "❌"
                        overall = "✅ SAFE" if (p["has_helmet"] and p["has_jacket"]) else "⚠️ UNSAFE"
                        st.text(f"Person {i}: Helmet {helmet} | Vest {jacket} → {overall}")
                else: st.info("No persons detected to analyze individually.")
            else: st.warning("No predictions could be made.")

# ----------------- LIVE WEBCAM TAB -----------------
with tab2:
    st.header("Live Webcam PPE Detection")
    with st.sidebar:
        st.divider(); st.subheader("Webcam Settings")
        DETECT_INTERVAL = st.number_input("Detect every N frames", 1, 60, 6, help="Higher number reduces API calls.")
        RENDER_SKIP = st.number_input("Render every N frames", 1, 10, 2, help="Higher number reduces UI lag.")
    if "run_webcam" not in st.session_state: st.session_state.run_webcam = False
    def toggle_webcam(): st.session_state.run_webcam = not st.session_state.run_webcam
    st.button("Toggle Webcam Feed", on_click=toggle_webcam, type="primary")
    if st.session_state.run_webcam:
        FRAME_WINDOW, dashboard_cols = st.image([]), st.columns(3)
        METRIC_PERSONS, METRIC_SAFE, METRIC_UNSAFE = dashboard_cols[0].empty(), dashboard_cols[1].empty(), dashboard_cols[2].empty()
        CHART_PLACEHOLDER, FPS_PLACEHOLDER, ALERT_PLACEHOLDER = st.empty(), st.empty(), st.empty()
        predictions_store = {"preds": None, "running": False, "last_request_time": 0}
        pred_lock, fps_values = threading.Lock(), []
        def detection_thread_fn(frame_for_detection):
            with pred_lock:
                if predictions_store["running"]: return
                predictions_store["running"] = True
            preds = detect_ppe(frame_for_detection)
            with pred_lock:
                predictions_store["preds"], predictions_store["running"], predictions_store["last_request_time"] = preds, False, time.time()
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened(): st.error("Cannot open webcam."); st.session_state.run_webcam = False; st.stop()
            frame_idx, prev_time, last_preds_for_overlay = 0, time.time(), None
            while st.session_state.run_webcam:
                ret, frame = cap.read()
                if not ret: st.warning("Failed to grab frame."); break
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT)); display_frame = frame.copy()
                with pred_lock: is_busy = predictions_store["running"]
                if (frame_idx % DETECT_INTERVAL == 0) and not is_busy:
                    threading.Thread(target=detection_thread_fn, args=(frame.copy(),), daemon=True).start()
                with pred_lock: preds_json = predictions_store.get("preds")
                if preds_json:
                    annotated, preds_list, _ = draw_predictions_on_frame(display_frame.copy(), preds_json)
                    last_preds_for_overlay = annotated
                    total_persons, safe_count, unsafe_count, _ = compute_person_ppe_stats(preds_list)
                    METRIC_PERSONS.metric("Total Persons", total_persons); METRIC_SAFE.metric("✅ Safe", safe_count); METRIC_UNSAFE.metric("❌ Unsafe", unsafe_count)
                    fig = plot_ppe_pie({"Safe": safe_count, "Unsafe": unsafe_count}); CHART_PLACEHOLDER.pyplot(fig); plt.close(fig)
                    display_frame = annotated
                    if unsafe_count > 0:
                        alert_message = f"⚠️ ALERT: {unsafe_count} unsafe person(s) detected!"
                        ALERT_PLACEHOLDER.warning(alert_message); send_email_alert(subject="[ACTION REQUIRED] PPE Violation Alert", body=alert_message)
                    else: ALERT_PLACEHOLDER.empty()
                elif last_preds_for_overlay is not None: display_frame = last_preds_for_overlay
                now = time.time(); fps = 1.0 / max(1e-6, (now - prev_time)); prev_time = now
                fps_values.append(fps);
                if len(fps_values) > 10: fps_values.pop(0)
                avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0
                FPS_PLACEHOLDER.caption(f"Stream FPS: {avg_fps:.1f}")
                if frame_idx % RENDER_SKIP == 0: FRAME_WINDOW.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                frame_idx += 1
        except Exception as e: st.error(f"Webcam stream error: {e}")
        finally:
            if 'cap' in locals() and cap.isOpened(): cap.release()
            st.session_state.run_webcam = False
    else: st.info("Click the button above to start the live webcam feed.")

# ----------------- VIDEO UPLOAD TAB -----------------
with tab3:
    st.header("Upload a Video for Frame-by-Frame Analysis")
    uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_video:
        # Use a temporary file to store the uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        # Display the uploaded video
        st.video(video_path)
        
        if st.button("Analyze Video", type="primary"):
            analysis_results = []
            results_table = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                for frame_num in range(total_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Perform detection on the frame
                    predictions = detect_ppe(frame)
                    if predictions:
                        _, preds_list, _ = draw_predictions_on_frame(frame, predictions) # No need for annotated frame here
                        total_p, safe_c, unsafe_c, _ = compute_person_ppe_stats(preds_list)
                        
                        # Store results
                        analysis_results.append({
                            'Frame': frame_num + 1,
                            'Total Persons': total_p,
                            'Safe Count': safe_c,
                            'Unsafe Count': unsafe_c
                        })
                    else: # Handle case where API call fails for a frame
                         analysis_results.append({
                            'Frame': frame_num + 1, 'Total Persons': 'N/A', 'Safe Count': 'N/A', 'Unsafe Count': 'N/A'
                        })
                    
                    # Update UI in real-time
                    progress = (frame_num + 1) / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_num + 1}/{total_frames}...")
                    
                    # Display the results table, updating it with each frame
                    results_df = pd.DataFrame(analysis_results).set_index('Frame')
                    results_table.dataframe(results_df)

                cap.release()
                status_text.success("✅ Analysis complete! Generating annotated video...")

                # --- Second pass: Create annotated video ---
                annotated_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                cap = cv2.VideoCapture(video_path) # Re-open the video
                
                # Get video properties for VideoWriter
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (frame_width, frame_height))

                for frame_num in range(total_frames):
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # We need to re-run detection to draw boxes
                    predictions = detect_ppe(frame)
                    annotated_frame, _, _ = draw_predictions_on_frame(frame, predictions)
                    out.write(annotated_frame)
                    progress_bar.progress((frame_num + 1) / total_frames)

                cap.release()
                out.release()
                
                status_text.success("✅ Annotated video generated successfully!")
                progress_bar.empty()
                
                st.subheader("🎬 Annotated Video")
                st.video(annotated_video_path)
                
                with open(annotated_video_path, "rb") as file:
                    st.download_button(
                        label="Download Annotated Video",
                        data=file,
                        file_name="annotated_video.mp4",
                        mime="video/mp4"
                    )

            except Exception as e:
                st.error(f"An error occurred during video processing: {e}")
            finally:
                # Clean up the temporary files
                os.remove(video_path)
                if 'annotated_video_path' in locals() and os.path.exists(annotated_video_path):
                     os.remove(annotated_video_path)