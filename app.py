import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Heart Attack Risk Predictor", page_icon="❤️", layout="wide")

# Load models safely
@st.cache_resource
def load_assets():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    return model, scaler, features

try:
    model, scaler, features = load_assets()
except:
    st.error("❌ Model assets missing! Please run 'python model.py' in terminal first.")

# Custom CSS for Professional Dark UI
st.markdown("""
    <style>
    .main { background-color: #0f111a; }
    .stButton>button {
        background: linear-gradient(135deg, #ff4b4b, #ff7676);
        color: white; font-weight: bold; border: none;
        border-radius: 8px; padding: 12px; transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 4px 15px rgba(255,75,75,0.4); }
    .card {
        background-color: #1a1c24; padding: 20px;
        border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Title Header
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>❤️ Advanced Heart Attack Risk Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8a8f98;'>Predictive Health Modelling using Gradient Boosting Intelligence</p>", unsafe_allow_html=True)
st.write("---")

# Layout Columns
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='card'><h3>📋 Patient Demographics & Clinical Metrics</h3></div>", unsafe_allow_html=True)
    
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        age = st.slider("Age (Umar)", 1, 100, 45)
        sex = st.selectbox("Gender / Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", value=130)
        cholesterol = st.number_input("Serum Cholesterol (mg/dl)", value=220)
    with sub_col2:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "True (1)" if x==1 else "False (0)")
        resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate Achieved (MHR)", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])
        oldpeak = st.number_input("ST Depression (Oldpeak)", value=1.0, step=0.1)
        st_slope = st.selectbox("Peak Exercise ST Slope", ["Up", "Flat", "Down"])

with col2:
    st.markdown("<div class='card'><h3>🎯 Diagnostic Output</h3></div>", unsafe_allow_html=True)
    st.write("")
    
    # Mapping inputs to match training data
    mapping = {
        'Sex': {'M': 1, 'F': 0},
        'ChestPainType': {'ATA': 1, 'NAP': 2, 'ASY': 0, 'TA': 3},
        'RestingECG': {'Normal': 1, 'ST': 2, 'LVH': 0},
        'ExerciseAngina': {'N': 0, 'Y': 1},
        'ST_Slope': {'Up': 2, 'Flat': 1, 'Down': 0}
    }
    
    input_data = pd.DataFrame([{
        'Age': age, 'Sex': mapping['Sex'][sex], 'ChestPainType': mapping['ChestPainType'][chest_pain],
        'RestingBP': resting_bp, 'Cholesterol': cholesterol, 'FastingBS': fasting_bs,
        'RestingECG': mapping['RestingECG'][resting_ecg], 'MaxHR': max_hr,
        'ExerciseAngina': mapping['ExerciseAngina'][exercise_angina], 'Oldpeak': oldpeak, 'ST_Slope': mapping['ST_Slope'][st_slope]
    }])

    if st.button("Run Clinical Assessment 🚀", use_container_width=True):
        # Scale and Predict
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)[0][1]

        # Circular Gauge Chart for Professional Look
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Probability %", 'font': {'size': 18, 'color': 'white'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#ff4b4b" if prediction[0] == 1 else "#2ebd59"},
                'bgcolor': "#1a1c24",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(46, 189, 89, 0.2)'},
                    {'range': [40, 70], 'color': 'rgba(255, 165, 0, 0.2)'},
                    {'range': [70, 100], 'color': 'rgba(255, 75, 75, 0.2)'}
                ],
            }
        ))
        fig.update_layout(paper_bgcolor="#0f111a", plot_bgcolor="#0f111a", font={'color': "white"}, height=250, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

        if prediction[0] == 1:
            st.error(f"⚠️ **HIGH RISK DETECTED:** Clinical markers strongly correlate with coronary artery disease patterns.")
        else:
            st.success(f"✅ **LOW RISK DETECTED:** Cardiovascular metrics fall within standard healthy thresholds.")