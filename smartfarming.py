import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import cv2
from PIL import Image
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="AI Farming Assistant",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)

class CropRecommender:
    def __init__(self):
        # Initialize with some sample data - in production, you'd load a trained model
        self.soil_types = ['clay', 'loamy', 'sandy']
        self.seasons = ['summer', 'winter', 'monsoon']
        
    def predict(self, n, p, k, temperature, humidity, ph, rainfall, soil_type, season):
        # Simplified logic - in production, use a trained ML model
        if ph < 5.5:
            return "Rice"
        elif temperature > 30 and rainfall < 100:
            return "Cotton"
        else:
            return "Wheat"

class DiseaseDetector:
    def __init__(self):
        # In production, load a trained deep learning model
        pass
    
    def predict(self, image):
        # Simplified logic - in production, use a trained deep learning model
        # This is just a placeholder implementation
        img_array = np.array(image)
        avg_color = np.mean(img_array, axis=(0, 1))
        
        # Simple rule-based detection
        if avg_color[1] < 100:  # If green component is low
            return "Possible leaf blight detected"
        else:
            return "Plant appears healthy"

class YieldPredictor:
    def __init__(self):
        # Initialize with sample data - in production, load a trained model
        pass
    
    def predict(self, crop_type, area, fertilizer, rainfall):
        # Simplified prediction logic
        base_yield = {
            'Rice': 4000,
            'Wheat': 3000,
            'Cotton': 2000
        }
        
        estimated_yield = base_yield.get(crop_type, 2500) * (area / 100)
        # Apply simple multipliers based on inputs
        estimated_yield *= (1 + fertilizer/100)
        estimated_yield *= (1 + rainfall/1000)
        
        return round(estimated_yield, 2)

def main():
    st.title("🌾 AI Farming Assistant")
    
    # Initialize our components
    crop_recommender = CropRecommender()
    disease_detector = DiseaseDetector()
    yield_predictor = YieldPredictor()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a Feature",
        ["Crop Recommendation", "Disease Detection", "Yield Prediction"]
    )
    
    if page == "Crop Recommendation":
        st.header("🌱 Crop Recommendation System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n = st.number_input("Nitrogen (N) content", 0, 140, 50)
            p = st.number_input("Phosphorous (P) content", 0, 140, 50)
            k = st.number_input("Potassium (K) content", 0, 200, 50)
            temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
        
        with col2:
            humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
            ph = st.number_input("pH level", 0.0, 14.0, 7.0)
            rainfall = st.number_input("Rainfall (mm)", 0.0, 1000.0, 200.0)
            soil_type = st.selectbox("Soil Type", crop_recommender.soil_types)
            season = st.selectbox("Season", crop_recommender.seasons)
        
        if st.button("Get Recommendation"):
            result = crop_recommender.predict(n, p, k, temperature, humidity, ph, rainfall, soil_type, season)
            st.success(f"Based on the provided conditions, the recommended crop is: {result}")
            
            # Display additional insights
            st.subheader("Soil Analysis")
            fig = px.bar(
                x=['Nitrogen', 'Phosphorous', 'Potassium'],
                y=[n, p, k],
                labels={'x': 'Nutrient', 'y': 'Content Level'}
            )
            st.plotly_chart(fig)
    
    elif page == "Disease Detection":
        st.header("🔍 Plant Disease Detection")
        
        uploaded_file = st.file_uploader("Upload an image of the plant leaf", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Detect Disease"):
                result = disease_detector.predict(image)
                st.write("Analysis Result:")
                st.info(result)
                
                # Display confidence scores (in production, these would come from the model)
                st.subheader("Detection Confidence")
                confidence_data = {
                    'Healthy': 0.8,
                    'Leaf Blight': 0.15,
                    'Other': 0.05
                }
                fig = px.pie(values=list(confidence_data.values()), names=list(confidence_data.keys()))
                st.plotly_chart(fig)
    
    else:  # Yield Prediction
        st.header("📊 Crop Yield Prediction")
        
        crop_type = st.selectbox("Select Crop", ['Rice', 'Wheat', 'Cotton'])
        area = st.number_input("Area (hectares)", 1, 1000, 100)
        fertilizer = st.slider("Fertilizer Usage (kg/ha)", 0, 500, 200)
        rainfall = st.number_input("Expected Rainfall (mm)", 0.0, 2000.0, 500.0)
        
        if st.button("Predict Yield"):
            predicted_yield = yield_predictor.predict(crop_type, area, fertilizer, rainfall)
            st.success(f"Predicted yield: {predicted_yield:.2f} kg")
            
            # Display historical comparison (dummy data)
            st.subheader("Historical Yield Comparison")
            historical_data = pd.DataFrame({
                'Year': [2020, 2021, 2022, 2023],
                'Yield': [predicted_yield * 0.9, predicted_yield * 0.95, 
                         predicted_yield * 1.05, predicted_yield]
            })
            fig = px.line(historical_data, x='Year', y='Yield', 
                         title=f'Historical {crop_type} Yield Trends')
            st.plotly_chart(fig)
            
            # Display factors affecting yield
            st.subheader("Factors Affecting Yield")
            factors = pd.DataFrame({
                'Factor': ['Area', 'Fertilizer', 'Rainfall'],
                'Impact': [area/100, fertilizer/200, rainfall/500]
            })
            fig = px.bar(factors, x='Factor', y='Impact',
                        title='Relative Impact of Different Factors')
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()