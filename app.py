from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Enable CORS for frontend communication

# Global variables for model and scaler
model = None
scaler = None

def load_models():
    """Load the trained model and scaler"""
    global model, scaler
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        print("Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Load models on startup
models_loaded = load_models()

@app.route('/', methods=['GET'])
def serve_home():
    """Serve the frontend"""
    return send_file('index.html', mimetype='text/html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint for heart disease prediction
    Expects JSON with patient data
    """
    if not models_loaded:
        return jsonify({
            'error': 'Models not loaded. Please ensure heart_disease_model.pkl and scaler.pkl are present.'
        }), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Define required fields in exact training order
        required_fields = [
            'Age', 'Sex', 'Chest_Pain_Type', 'Resting_Blood_Pressure',
            'Cholesterol', 'Fasting_Blood_Sugar', 'Resting_ECG',
            'Max_Heart_Rate', 'Exercise_Angina', 'ST_Depression', 'ST_Slope'
        ]
        
        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([data], columns=required_fields)
        
        # Convert to numeric and handle any errors
        try:
            input_df = input_df.astype(float)
        except ValueError as e:
            return jsonify({'error': f'Invalid numeric values: {str(e)}'}), 400
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Format result
        has_disease = bool(prediction == 1)
        confidence = float(probabilities[prediction] * 100)
        
        # Prepare response
        response = {
            'success': True,
            'prediction': 'Heart Disease Detected' if has_disease else 'No Heart Disease Detected',
            'has_disease': has_disease,
            'confidence': round(confidence, 2),
            'probability_no_disease': round(float(probabilities[0] * 100), 2),
            'probability_disease': round(float(probabilities[1] * 100), 2),
            'risk_level': get_risk_level(confidence, has_disease)
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

def get_risk_level(confidence, has_disease):
    """Determine risk level based on prediction and confidence"""
    if not has_disease:
        return 'Low Risk'
    elif confidence >= 80:
        return 'High Risk'
    elif confidence >= 60:
        return 'Moderate Risk'
    else:
        return 'Low-Moderate Risk'

@app.route('/api/validate', methods=['POST'])
def validate_input():
    """Validate input data without making prediction"""
    try:
        data = request.get_json()
        
        required_fields = [
            'Age', 'Sex', 'Chest_Pain_Type', 'Resting_Blood_Pressure',
            'Cholesterol', 'Fasting_Blood_Sugar', 'Resting_ECG',
            'Max_Heart_Rate', 'Exercise_Angina', 'ST_Depression', 'ST_Slope'
        ]
        
        validation_errors = []
        
        for field in required_fields:
            if field not in data:
                validation_errors.append(f'{field} is required')
            else:
                try:
                    float(data[field])
                except (ValueError, TypeError):
                    validation_errors.append(f'{field} must be a valid number')
        
        if validation_errors:
            return jsonify({
                'valid': False,
                'errors': validation_errors
            }), 400
        
        return jsonify({
            'valid': True,
            'message': 'All inputs are valid'
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Validation error: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("Starting Heart Disease Prediction API...")
    print(f"Models loaded: {models_loaded}")
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    print(f"Server running on port {port}")
    app.run(debug=debug, host='0.0.0.0', port=port)