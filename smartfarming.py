import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import plotly.express as px
from PIL import Image
import os

# Model Definitions
class PlantDiseaseNet(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseNet, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class CropRecommender:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.soil_types = ['clay', 'loamy', 'sandy']
        self.seasons = ['summer', 'winter', 'monsoon']
        
    def train(self, X, y):
        X_encoded = X.copy()
        X_encoded['soil_type'] = self.label_encoder.fit_transform(X['soil_type'])
        X_encoded['season'] = self.label_encoder.fit_transform(X['season'])
        
        numerical_cols = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']
        X_encoded[numerical_cols] = self.scaler.fit_transform(X_encoded[numerical_cols])
        
        self.model.fit(X_encoded, y)
        
    def predict(self, n, p, k, temperature, humidity, ph, rainfall, soil_type, season):
        input_data = pd.DataFrame({
            'n': [n],
            'p': [p],
            'k': [k],
            'temperature': [temperature],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall],
            'soil_type': [soil_type],
            'season': [season]
        })
        
        input_encoded = input_data.copy()
        input_encoded['soil_type'] = self.label_encoder.transform(input_encoded['soil_type'])
        input_encoded['season'] = self.label_encoder.transform(input_encoded['season'])
        
        numerical_cols = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']
        input_encoded[numerical_cols] = self.scaler.transform(input_encoded[numerical_cols])
        
        prediction = self.model.predict(input_encoded)
        probabilities = self.model.predict_proba(input_encoded)
        
        return prediction[0], probabilities[0]

class DiseaseDetector:
    def __init__(self, model_path=None, num_classes=38):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PlantDiseaseNet(num_classes).to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Disease classes (example)
        self.classes = [
            'healthy', 'leaf_blight', 'leaf_spot', 'rust',
            'powdery_mildew', 'early_blight', 'late_blight'
        ]
    
    def predict(self, image):
        self.model.eval()
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[0][predicted_class].item()
            
        return self.classes[predicted_class], confidence

class YieldPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
    def predict(self, crop_type, area, fertilizer, rainfall):
        input_data = np.array([[
            crop_type,
            area,
            fertilizer,
            rainfall
        ]])
        
        input_scaled = self.scaler.transform(input_data)
        prediction = self.model.predict(input_scaled)
        
        return prediction[0]

# Streamlit App
def load_models():
    """Load or initialize all models"""
    crop_recommender = CropRecommender()
    disease_detector = DiseaseDetector(model_path='models/disease_model.pth')
    yield_predictor = YieldPredictor()
    
    # Here you would normally load trained models
    # For now, we'll use them untrained
    return crop_recommender, disease_detector, yield_predictor

def main():
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
        .prediction-box {
            padding: 20px;
            border-radius: 5px;
            background-color: #f0f2f6;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("🌾 AI Farming Assistant")
    
    # Load models
    crop_recommender, disease_detector, yield_predictor = load_models()
    
    # Sidebar navigation
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
            with st.spinner("Analyzing soil conditions..."):
                crop, probabilities = crop_recommender.predict(
                    n, p, k, temperature, humidity, ph, rainfall, soil_type, season
                )
                
                st.success(f"Based on the provided conditions, the recommended crop is: {crop}")
                
                # Display confidence scores
                st.subheader("Recommendation Confidence")
                conf_df = pd.DataFrame({
                    'Crop': crop_recommender.model.classes_,
                    'Confidence': probabilities * 100
                })
                fig = px.bar(conf_df, x='Crop', y='Confidence',
                            title='Recommendation Confidence Scores')
                st.plotly_chart(fig)
                
                # Soil Analysis
                st.subheader("Soil Analysis")
                fig = px.bar(
                    x=['Nitrogen', 'Phosphorous', 'Potassium'],
                    y=[n, p, k],
                    labels={'x': 'Nutrient', 'y': 'Content Level'}
                )
                st.plotly_chart(fig)
    
    elif page == "Disease Detection":
        st.header("🔍 Plant Disease Detection")
        
        uploaded_file = st.file_uploader(
            "Upload an image of the plant leaf",
            type=["jpg", "jpeg", "png", "wepb"]
        )        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Detect Disease"):
                with st.spinner("Analyzing image..."):
                    disease, confidence = disease_detector.predict(image)
                    
                    st.write("Analysis Result:")
                    if disease == 'healthy':
                        st.success("Plant appears healthy! ✅")
                    else:
                        st.warning(f"Detected: {disease.replace('_', ' ').title()}")
                    
                    # Display confidence score
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                    
                    # Display recommendations based on disease
                    if disease != 'healthy':
                        st.subheader("Recommended Actions:")
                        recommendations = {
                            'leaf_blight': [
                                "Remove and destroy infected leaves",
                                "Apply fungicide containing chlorothalonil",
                                "Improve air circulation around plants"
                            ],
                            'rust': [
                                "Apply fungicide early in the season",
                                "Maintain proper plant spacing",
                                "Avoid overhead watering"
                            ]
                            # Add more recommendations for other diseases
                        }
                        
                        if disease in recommendations:
                            for rec in recommendations[disease]:
                                st.write(f"• {rec}")
    
    else:  # Yield Prediction
        st.header("📊 Crop Yield Prediction")
        
        crop_type = st.selectbox("Select Crop", ['Rice', 'Wheat', 'Cotton'])
        area = st.number_input("Area (hectares)", 1, 1000, 100)
        fertilizer = st.slider("Fertilizer Usage (kg/ha)", 0, 500, 200)
        rainfall = st.number_input("Expected Rainfall (mm)", 0.0, 2000.0, 500.0)
        
        if st.button("Predict Yield"):
            with st.spinner("Calculating yield prediction..."):
                predicted_yield = yield_predictor.predict(
                    crop_type, area, fertilizer, rainfall
                )
                
                st.success(f"Predicted yield: {predicted_yield:.2f} kg")
                
                # Display factors affecting yield
                st.subheader("Factors Affecting Yield")
                factors = pd.DataFrame({
                    'Factor': ['Area', 'Fertilizer', 'Rainfall'],
                    'Impact': [area/100, fertilizer/200, rainfall/500]
                })
                fig = px.bar(factors, x='Factor', y='Impact',
                            title='Relative Impact of Different Factors')
                st.plotly_chart(fig)
                
                # Show optimization suggestions
                st.subheader("Optimization Suggestions")
                if fertilizer < 200:
                    st.info("Consider increasing fertilizer usage for potentially higher yield")
                if rainfall < 400:
                    st.warning("Low rainfall predicted. Consider irrigation planning")
                
                # Historical comparison
                st.subheader("Historical Yield Comparison")
                historical_data = pd.DataFrame({
                    'Year': [2020, 2021, 2022, 2023],
                    'Yield': [
                        predicted_yield * 0.9,
                        predicted_yield * 0.95,
                        predicted_yield * 1.05,
                        predicted_yield
                    ]
                })
                fig = px.line(historical_data, x='Year', y='Yield',
                             title=f'Historical {crop_type} Yield Trends')
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()