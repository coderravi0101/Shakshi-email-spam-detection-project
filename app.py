import os
import logging
import joblib
import numpy as np
from flask import Flask, request, render_template, jsonify
from datetime import datetime

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] in %(module)s: %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

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
            logger.info("✅ Machine Learning artifacts successfully loaded.")
        else:
            logger.warning(f"⚠️ Artifact files not found. Expected paths:")
            logger.warning(f"   - Model: {MODEL_PATH}")
            logger.warning(f"   - Vectorizer: {VECTORIZER_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to load artifacts: {str(e)}", exc_info=True)

# Initial Artifact Load
load_artifacts()

@app.route('/', methods=['GET'])
def index():
    """Renders main user interface."""
    logger.info("📄 Serving index page")
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Service Health Probe for Load Balancers / Kubernetes."""
    status = "healthy" if (model and vectorizer) else "degraded"
    code = 200 if status == "healthy" else 503
    health_data = {
        "status": status,
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "timestamp": datetime.utcnow().isoformat()
    }
    logger.info(f"🔍 Health check: {status}")
    return jsonify(health_data), code

@app.route('/api/predict', methods=['POST'])
def predict():
    """Unified API Endpoint for web interface and REST consumers."""
    
    # Validate Model Availability
    if not model or not vectorizer:
        logger.error("❌ Model engine unavailable")
        return jsonify({
            'status': 'error',
            'message': 'Model engine is unavailable. Check system logs.',
            'timestamp': datetime.utcnow().isoformat()
        }), 503

    # Parse JSON Payload
    try:
        data = request.get_json(force=True)
    except Exception as e:
        logger.warning(f"⚠️ Invalid JSON payload: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Invalid JSON payload.',
            'timestamp': datetime.utcnow().isoformat()
        }), 400

    # Extract and Validate Message
    message = data.get('message', '').strip() if data else ''

    if not message:
        logger.warning("⚠️ Empty message provided")
        return jsonify({
            'status': 'error',
            'message': 'Payload field "message" cannot be empty.',
            'timestamp': datetime.utcnow().isoformat()
        }), 422

    # Perform Inference
    try:
        logger.info(f"📊 Processing email (length: {len(message)} chars)")
        
        # Transform & Predict
        transformed_text = vectorizer.transform([message])
        prediction_raw = model.predict(transformed_text)[0]
        probabilities = model.predict_proba(transformed_text)[0]
        confidence = round(float(np.max(probabilities)) * 100, 2)

        # Standardize Output Label
        prediction_label = "Spam" if str(prediction_raw).lower() in ['1', 'spam', 'true'] else "Ham"

        logger.info(f"✅ Prediction: {prediction_label} (Confidence: {confidence}%)")

        return jsonify({
            'status': 'success',
            'prediction': prediction_label,
            'confidence': confidence,
            'message': message[:100] + '...' if len(message) > 100 else message,
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"❌ Inference error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Internal inference failure.',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    logger.warning(f"⚠️ Resource not found: {request.path}")
    return jsonify({
        'status': 'error',
        'message': 'Resource not found.',
        'timestamp': datetime.utcnow().isoformat()
    }), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    logger.error(f"❌ Server error: {str(error)}", exc_info=True)
    return jsonify({
        'status': 'error',
        'message': 'Internal server error.',
        'timestamp': datetime.utcnow().isoformat()
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🚀 Starting Spam Detection Service...")
    logger.info(f"📍 Server: http://0.0.0.0:{port}")
    logger.info(f"🔧 Debug Mode: {debug_mode}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
