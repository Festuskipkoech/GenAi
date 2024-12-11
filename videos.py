from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from tempfile import NamedTemporaryFile
import shutil
from PIL import Image
import cv2
import numpy as np
from typing import List

# Load environment variables
load_dotenv()

# Configure Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_frames(video_path: str, max_frames: int = 10) -> List[Image.Image]:
    """
    Extract frames from the video file and convert them to PIL Images.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract
    
    Returns:
        List of PIL Image objects
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // max_frames)
    
    frames = []
    frame_count = 0
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            frames.append(pil_image)
            
        frame_count += 1
    
    cap.release()
    return frames

def get_gemini_response(input_text: str, video_path: str) -> str:
    """
    Process video frames with the Gemini API.
    
    Args:
        input_text: User's input prompt
        video_path: Path to the video file
    
    Returns:
        Generated response text
    """
    try:
        # Extract frames from the video
        frames = extract_frames(video_path)
        
        if not frames:
            raise ValueError("No frames could be extracted from the video")
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create the prompt with both text and images
        prompt_parts = [input_text] + frames
        
        # Generate content using the model
        response = model.generate_content(prompt_parts)
        return response.text
    
    except Exception as e:
        return f"Error processing video: {str(e)}"

# Initialize the Streamlit app
st.header("Gemini Video Processing")

# Add description
st.markdown("""
This app processes videos using Google's Gemini Pro Vision model. 
- Upload a video file (MP4, MOV, or AVI format)
- Enter your prompt/question about the video
- The app will extract key frames and analyze them using Gemini
""")

# Text input
input_text = st.text_input("Input prompt:", 
                          placeholder="What's happening in this video?",
                          key="input_text")

# File uploader for video files
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

# Display the uploaded video
if uploaded_file is not None:
    st.video(uploaded_file)

# Button to submit the query
submit = st.button("Submit query")

# Handle submission
if submit:
    if uploaded_file is None:
        st.error("Please upload a video file")
    elif not input_text.strip():
        st.error("Please enter a prompt")
    else:
        with st.spinner("Processing video..."):
            # Save the video to a temporary file
            with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                shutil.copyfileobj(uploaded_file, temp_file)
                temp_video_path = temp_file.name
            
            try:
                # Get response from the Gemini API
                response = get_gemini_response(input_text, temp_video_path)
                
                # Display the response
                st.subheader("Gemini API Response")
                st.write(response)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)