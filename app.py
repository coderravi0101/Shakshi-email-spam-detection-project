import os
import logging
import joblib
import numpy as np
from flask import Flask, request, render_template, jsonify

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] in %(module)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define Artifact Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.joblib')
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, 'vectorizer.joblib')

model = None
vectorizer = None

def load_artifacts():
    """Loads Machine Learning artifacts into memory safely."""
    global model, vectorizer
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            logger.info("Machine Learning artifacts successfully loaded.")
        else:
            logger.warning("Artifact files not found in 'artifacts/' directory.")
    except Exception as e:
        logger.error(f"Failed to load artifacts: {str(e)}", exc_info=True)

# Initial Artifact Load
load_artifacts()

@app.route('/', methods=['GET'])
def index():
    """Renders main user interface."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Service Health Probe for Load Balancers / Kubernetes."""
    status = "healthy" if (model and vectorizer) else "degraded"
    code = 200 if status == "healthy" else 503
    return jsonify({"status": status, "model_loaded": model is not None}), code

@app.route('/api/predict', methods=['POST'])
def predict():
    """Unified API Endpoint for web interface and REST consumers."""
    if not model or not vectorizer:
        return jsonify({
            'status': 'error',
            'message': 'Model engine is unavailable. Check system logs.'
        }), 503

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'status': 'error', 'message': 'Invalid JSON payload.'}), 400

    message = data.get('message', '').strip() if data else ''

    if not message:
        return jsonify({'status': 'error', 'message': 'Payload field "message" cannot be empty.'}), 422

    try:
        # Transform & Predict
        transformed_text = vectorizer.transform([message])
        prediction_raw = model.predict(transformed_text)[0]
        probabilities = model.predict_proba(transformed_text)[0]
        confidence = round(float(np.max(probabilities)) * 100, 2)

        # Standardize Output Label
        prediction_label = "Spam" if str(prediction_raw).lower() in ['1', 'spam', 'true'] else "Ham"

        return jsonify({
            'status': 'success',
            'prediction': prediction_label,
            'confidence': confidence,
            'message': message
        }), 200

    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'Internal inference failure.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
