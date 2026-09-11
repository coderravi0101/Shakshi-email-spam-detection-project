import os
import logging
import joblib
import numpy as np
import streamlit as st
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Spam Detection Engine",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] in %(module)s: %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define Artifact Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.joblib')
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, 'vectorizer.joblib')

@st.cache_resource
def load_artifacts():
    """Loads and caches Machine Learning artifacts into memory safely."""
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            logger.info("✅ Machine Learning artifacts successfully loaded.")
            return model, vectorizer, True
        else:
            logger.warning(f"⚠️ Artifact files not found in 'artifacts/' directory.")
            return None, None, False
    except Exception as e:
        logger.error(f"❌ Failed to load artifacts: {str(e)}", exc_info=True)
        return None, None, False

# Load ML Models
model, vectorizer, artifacts_loaded = load_artifacts()

# Sidebar Setup
with st.sidebar:
    st.header("⚙️ System Status")
    if artifacts_loaded:
        st.success("Model Engine: Active")
        st.caption("TF-IDF Naive Bayes Engine")
    else:
        st.error("Model Engine: Offline")
        st.caption("Missing artifact files in `/artifacts` folder.")
    
    st.divider()
    st.markdown("### 👨‍💻 Developer Info")
    st.info("**Created by Shashi**")
    st.caption("Enterprise NLP & Fraud Detection System")

# Main Header
st.title("🛡️ Text Classification System")
st.subheader("Enterprise Spam & Fraud Detection Engine")
st.caption("Analyze suspicious messages, emails, or texts in real time.")

st.divider()

# Input Section
user_message = st.text_area(
    label="Enter Text / Message Content:",
    placeholder="e.g., Congratulations! You have won a free $1000 gift card. Click here to claim your reward now...",
    height=150
)

# Inference Action
if st.button("🚀 Analyze Message", use_container_width=True, type="primary"):
    if not artifacts_loaded:
        st.error("❌ Model engine unavailable. Please check system artifacts.")
        logger.error("❌ Prediction attempted while artifacts were missing.")
    elif not user_message.strip():
        st.warning("⚠️ Message content cannot be empty. Please enter text to analyze.")
        logger.warning("⚠️ Empty message input submitted.")
    else:
        try:
            logger.info(f"📊 Processing text input (Length: {len(user_message)} chars)")
            
            # Vectorize & Predict
            transformed_text = vectorizer.transform([user_message])
            prediction_raw = model.predict(transformed_text)[0]
            probabilities = model.predict_proba(transformed_text)[0]
            confidence = round(float(np.max(probabilities)) * 100, 2)

            # Standardize Label
            prediction_label = "Spam" if str(prediction_raw).lower() in ['1', 'spam', 'true'] else "Ham"
            
            logger.info(f"✅ Prediction: {prediction_label} (Confidence: {confidence}%)")

            st.divider()

            # Output UI Rendering
            if prediction_label == "Spam":
                st.error("🚨 **Spam / Fraud Detected**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Classification Result", value="Spam")
                with col2:
                    st.metric(label="Confidence Score", value=f"{confidence}%")
            else:
                st.success("✅ **Safe / Legitimate Text (Ham)**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Classification Result", value="Ham")
                with col2:
                    st.metric(label="Confidence Score", value=f"{confidence}%")

        except Exception as e:
            st.error("❌ Internal inference error occurred while processing text.")
            logger.error(f"❌ Inference error: {str(e)}", exc_info=True)

# Footer
st.divider()
st.caption("Powered by Scikit-Learn • Streamlit Engine • Created by Shashi")
