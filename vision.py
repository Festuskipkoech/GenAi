from dotenv import load_dotenv
# loading all env variables
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

# function to load gemini pro model and get responses
model=genai.GenerativeModel('gemini-1.5-flash')
def get_gemini_response(input, image):
    if input !="":
        response=model.generate_content([input, image])
    else:
        response=model.generate_content(image)
    return response.text
# initialize the streamlit app
st.header("Gemini app using streamlit")
input =st.text_input("Input prompt: ", key="input")

uploaded_file=st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
image=""
if uploaded_file !=None:
    image=Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

submit=st.button("Submit query")

# if submit is clicked
if submit: 
    response=get_gemini_response(input, image)
    st.subheader("What the image is about")
    st.write(response) 