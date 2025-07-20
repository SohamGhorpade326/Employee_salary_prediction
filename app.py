# FILE 2: app.py
# PURPOSE: The Flask web server that serves predictions.
# TO RUN: flask run
# =============================================================================
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model pipeline
# This pipeline includes both the preprocessor and the model
try:
    model_pipeline = joblib.load('model.joblib')
    print("Model pipeline loaded successfully.")
except FileNotFoundError:
    print("Error: 'model.joblib' not found. Please run model_training.py first.")
    model_pipeline = None

# Define the feature names in the correct order
# This must match the order of columns in the training data (excluding the target)
feature_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
    'marital-status', 'occupation', 'relationship', 'race', 'gender',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
]

@app.route('/')
def home():
    """Renders the main page of the web application."""
    # The 'index.html' file should be in a 'templates' folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives user input, makes a prediction, and returns it."""
    if model_pipeline is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get the JSON data sent from the frontend
        data = request.get_json()
        
        # Convert the incoming dictionary to a pandas DataFrame
        # The DataFrame must have the columns in the correct order
        input_data = pd.DataFrame([data], columns=feature_names)
        
        # Ensure correct data types for numerical columns
        for col in ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
             if col in input_data.columns:
                input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

        # Use the pipeline to make a prediction
        # The pipeline handles both preprocessing and prediction
        prediction = model_pipeline.predict(input_data)
        prediction_proba = model_pipeline.predict_proba(input_data)

        # Prepare the response
        result = {
            'prediction': 'More than $50K' if prediction[0] == 1 else 'Less than or equal to $50K',
            'confidence_score': f"{np.max(prediction_proba[0]) * 100:.2f}%"
        }
        
        return jsonify(result)

    except Exception as e:
        # Log the error for debugging and return a generic error message
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 400

if __name__ == '__main__':
    # This allows you to run the app directly using 'python app.py'
    # For production, it's better to use a WSGI server like Gunicorn
    app.run(debug=True)
