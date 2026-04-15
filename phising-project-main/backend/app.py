from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import tensorflow as tf
from feature_extract import extract_features
from suggestions import suggest_alternative
from message_classifier import predict_message
import os



app = Flask(__name__)
CORS(app)
MESSAGE_CONFIDENCE_THRESHOLD = 0.75

# Paths (ensure model & scaler files are here)
BASE_DIR = os.path.dirname(__file__)
NN_MODEL_PATH = os.path.join(BASE_DIR, "phishing_detection_nn.keras")
RF_MODEL_PATH = os.path.join(BASE_DIR, "phishing_detection_rf.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

print("\n=== Loading Models ===")

# Load neural network
try:
    nn_model = tf.keras.models.load_model(NN_MODEL_PATH)
    print("✅ Neural Network Loaded")
except Exception as e:
    nn_model = None
    print("❌ Neural Network load error:", e)

# Load random forest
try:
    rf_model = joblib.load(RF_MODEL_PATH)
    print("✅ Random Forest Loaded")
except Exception as e:
    rf_model = None
    print("❌ Random Forest load error:", e)

# Load scaler (for NN)
try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler Loaded")
except Exception as e:
    scaler = None
    print("❌ Scaler load error:", e)


@app.route("/")
def home():
    return "✅ Phishing Detection API Running"


@app.route("/predict", methods=["POST"])
def predict():
    # Expect JSON: {"url": "https://example.com"}
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Send JSON: {\"url\":\"https://example.com\"}"}), 400

    url = str(data["url"]).strip()

    # Extract features (must match training)
    try:
        features = extract_features(url)
    except Exception as e:
        return jsonify({"error": "Feature extraction failed", "detail": str(e)}), 500

    # Prepare inputs
    X_raw = np.array(features).reshape(1, -1)   # for RF (raw features)

    # Scale only for NN
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X_raw)
        except Exception as e:
            return jsonify({
                "error": "Scaler transform failed — feature mismatch",
                "details": str(e),
                "model_features": getattr(scaler, "n_features_in_", "unknown"),
                "received_features": X_raw.shape[1]
            }), 500
    else:
        X_scaled = X_raw

    # Neural Network prediction (uses scaled input)
    if nn_model is not None:
        try:
            prob = float(nn_model.predict(X_scaled, verbose=0)[0][0])
            nn_label = "Phishing" if prob > 0.5 else "Legitimate"
        except Exception as e:
            nn_label = "Error"
    else:
        nn_label = "Unknown"

    # Random Forest prediction (uses raw input)
    if rf_model is not None:
        try:
            pred = rf_model.predict(X_raw)[0]
            rf_label = "Phishing" if int(pred) == 1 else "Legitimate"
        except Exception as e:
            rf_label = "Error"
    else:
        rf_label = "Unknown"

    # Final decision: if either model says Phishing, mark Phishing; else Legitimate
    if nn_label == "Phishing" or rf_label == "Phishing":
        final_result = "Phishing"
    elif nn_label in ("Unknown", "Error") and rf_label in ("Unknown", "Error"):
        final_result = "Unknown"
    else:
        final_result = "Legitimate"

        # Suggest safe alternatives ONLY if phishing
    suggested_links = []
    if final_result == "Phishing":
        suggested_links = suggest_alternative(url)


    return jsonify({
        "url": url,
        "final_result": final_result,
        "neural_network": nn_label,
        "random_forest": rf_label,
        "suggested_links": suggested_links
    })


@app.route("/predict-message", methods=["POST"])
def predict_message_endpoint():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Send JSON: {\"message\":\"Your text here\"}"}), 400

    message = str(data["message"]).strip()
    if not message:
        return jsonify({"error": "message cannot be empty"}), 400

    try:
        stp_label, confidence = predict_message(message)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Message model inference failed", "detail": str(e)}), 500

    if confidence < MESSAGE_CONFIDENCE_THRESHOLD:
        final_result = "Unknown"
    else:
        final_result = "Phishing" if stp_label == "P" else "Legitimate"

    return jsonify({
        "message": message,
        "stp_label": stp_label,
        "confidence": confidence,
        "final_result": final_result,
        "threshold": MESSAGE_CONFIDENCE_THRESHOLD
    })



if __name__ == "__main__":
    # debug True ok for local dev; use proper WSGI for production
    app.run(host="0.0.0.0", port=5000, debug=True)
