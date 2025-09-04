import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
from huggingface_hub import hf_hub_download
import os
import streamlit as st
from huggingface_hub import hf_hub_url
import requests

st.set_page_config(page_title="Real-time YOLO Detection", layout="centered")
st.title("Real-time Shot Detection")
st.markdown("Upload a video and analyze the match!")
model_path = "models/best.pt"

if not os.path.exists(model_path):
    os.makedirs("models", exist_ok=True)
    with st.spinner("Model not found. Downloading from Hugging Face..."):
        url = hf_hub_url(
            repo_id="Arup-ai/badminton-shot-detection",
            filename="best.pt"
        )
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        progress_bar = st.progress(0)
        with open(model_path, 'wb') as f:
            for i, data in enumerate(response.iter_content(block_size)):
                f.write(data)
                progress = (i * block_size) / total_size
                progress_bar.progress(min(progress, 1.0))
    st.success("Model downloaded successfully!")





@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

def process_video_realtime(video_file, model_path):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    video_path = tfile.name
    
    model = load_model(model_path)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)
    frame_counter = st.empty()
    
    frame_num = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=False)
        frame_num += 1
        progress = frame_num / total_frames
        progress_bar.progress(progress)
        frame_counter.text(f"Frame {frame_num}/{total_frames}")
        
    cap.release()
    st.success("✨ Processing complete!")

with st.sidebar:
    st.header("Config")
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded_file is not None:
    if st.button("Start Real-time Detection", type="primary"):
        with st.spinner("Processing video in real-time..."):
            try:
                process_video_realtime(uploaded_file, model_path)
            except Exception as e:
                st.error(f"Oops! Something went wrong: {str(e)}")
else:
    st.info("👆 Upload a video file to get started!")

st.markdown("---")
st.markdown("Built by Arup :3")
