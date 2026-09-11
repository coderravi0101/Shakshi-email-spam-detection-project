import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

# Load Model and Vectorizer safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

model = None
vectorizer = None

def load_artifacts():
    global model, vectorizer
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        print("Model and Vectorizer loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

load_artifacts()

# HTML Template with modern UI design
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spam Classification Engine</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #2b3674;
            --secondary-color: #4318ff;
            --bg-color: #f4f7fe;
            --card-bg: #ffffff;
            --text-main: #1b2559;
            --text-muted: #a3edd9;
        }
        body {
            background-color: var(--bg-color);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .main-card {
            background: var(--card-bg);
            border-radius: 20px;
            box-shadow: 0px 18px 40px rgba(112, 144, 176, 0.12);
            padding: 40px;
            max-width: 750px;
            width: 100%;
            border: none;
        }
        .header-title {
            color: var(--primary-color);
            font-weight: 700;
            font-size: 1.8rem;
        }
        .subtitle {
            color: #a318ff;
            font-size: 0.95rem;
            margin-bottom: 25px;
        }
        .form-control {
            border-radius: 12px;
            border: 1.5px solid #e0e5f2;
            padding: 15px;
            font-size: 1rem;
            transition: all 0.2s ease-in-out;
        }
        .form-control:focus {
            border-color: var(--secondary-color);
            box-shadow: 0 0 0 0.2rem rgba(67, 24, 255, 0.15);
        }
        .btn-predict {
            background-color: var(--secondary-color);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 30px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            width: 100%;
        }
        .btn-predict:hover {
            background-color: #3311cc;
            color: white;
            transform: translateY(-2px);
        }
        .result-box {
            border-radius: 16px;
            padding: 20px;
            margin-top: 25px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .result-spam {
            background-color: #ffe5e5;
            border-left: 6px solid #e53e3e;
            color: #9b2c2c;
        }
        .result-ham {
            background-color: #e6fffa;
            border-left: 6px solid #319795;
            color: #234e52;
        }
        .badge-confidence {
            font-size: 1rem;
            padding: 8px 16px;
            border-radius: 20px;
        }
        .footer-text {
            font-size: 0.8rem;
            color: #a3bbd6;
            text-align: center;
            margin-top: 25px;
        }
    </style>
</head>
<body>

<div class="main-card">
    <div class="text-center">
        <h2 class="header-title"><i class="fa-solid fa-shield-halved text-primary me-2"></i>Text Classification System</h2>
        <p class="subtitle text-secondary">Enterprise NLP Spam & Fraud Detection Engine</p>
    </div>

    <form method="POST" action="/predict">
        <div class="mb-3">
            <label for="message" class="form-label fw-bold">Enter Text / Message Content:</label>
            <textarea class="form-control" id="message" name="message" rows="5" placeholder="e.g., Congratulations! You have won a free $1000 gift card. Click here to claim your reward now..." required>{{ user_input if user_input else '' }}</textarea>
        </div>
        <button type="submit" class="btn btn-predict"><i class="fa-solid fa-paper-plane me-2"></i> Analyze Message</button>
    </form>

    {% if prediction %}
    <div class="result-box {% if prediction == 'Spam' %}result-spam{% else %}result-ham{% endif %}">
        <div>
            <h4 class="mb-1 fw-bold">
                {% if prediction == 'Spam' %}
                    <i class="fa-solid fa-triangle-exclamation me-2"></i> Spam Detected
                {% else %}
                    <i class="fa-solid fa-circle-check me-2"></i> Safe / Legitimate Text (Ham)
                {% endif %}
            </h4>
            <p class="mb-0 text-muted small">Model output based on Naive Bayes TF-IDF Classification</p>
        </div>
        <div>
            <span class="badge {% if prediction == 'Spam' %}bg-danger{% else %}bg-success{% endif %} badge-confidence">
                Probability: {{ probability }}%
            </span>
        </div>
    </div>
    {% endif %}

    <div class="footer-text">
        Powered by Scikit-Learn • Flask REST API • Production Ready Architecture
    </div>
</div>

</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return render_template_string(HTML_TEMPLATE, prediction="Error", probability=0, user_input="Artifacts not loaded.")

    if request.is_json:
        data = request.get_json()
        message = data.get('message', '')
    else:
        message = request.form.get('message', '')

    if not message.strip():
        return render_template_string(HTML_TEMPLATE)

    # Transform input text using vectorizer
    transformed_text = vectorizer.transform([message])

    # Predict using trained Naive Bayes classifier
    prediction = model.predict(transformed_text)[0]
    probabilities = model.predict_proba(transformed_text)[0]
    confidence = round(float(np.max(probabilities)) * 100, 2)

    if request.is_json:
        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'confidence_percentage': confidence,
            'input_message': message
        })

    return render_template_string(
        HTML_TEMPLATE,
        prediction=prediction,
        probability=confidence,
        user_input=message
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for programmatic access."""
    if not model or not vectorizer:
        return jsonify({'status': 'error', 'message': 'Model artifacts not loaded.'}), 500

    data = request.get_json(force=True)
    message = data.get('message', '')

    if not message:
        return jsonify({'status': 'error', 'message': 'Field "message" is required.'}), 400

    transformed_text = vectorizer.transform([message])
    prediction = model.predict(transformed_text)[0]
    probabilities = model.predict_proba(transformed_text)[0]
    confidence = round(float(np.max(probabilities)) * 100, 2)

    return jsonify({
        'status': 'success',
        'prediction': str(prediction),
        'confidence': confidence,
        'message': message
    })

if __name__ == '__main__':
    # Enterprise ready server execution settings
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
