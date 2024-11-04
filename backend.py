import json
import boto3
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize S3 client and specify model location
s3 = boto3.client('s3')
bucket_name = 'mymodelaws'
model_key = 'modelaws.joblib'
download_path = '/tmp/modelaws.joblib'  # Path to temporarily store model on the server

# Download and load the model when the server starts
try:
    s3.download_file(bucket_name, model_key, download_path)
    model = joblib.load(download_path)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None  # To ensure the app fails if model loading fails

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is live'})

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model could not be loaded'}), 500
    
    try:
        # Parse JSON body from POST request
        data = request.get_json()
        input_data = [[
            data.get('Pregnancies', 0),
            data.get('Glucose', 0),
            data.get('BloodPressure', 0),
            data.get('SkinThickness', 0),
            data.get('Insulin', 0),
            data.get('BMI', 0.0),
            data.get('DiabetesPedigreeFunction', 0.0),
            data.get('Age', 0)
        ]]

        # Convert input to DataFrame
        input_df = pd.DataFrame(input_data, columns=[
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ])
        
        # Make a prediction
        prediction = model.predict(input_df)
        predicted_label = int(prediction[0])  # Binary classification: 0 or 1

        return jsonify({'Predicted Label': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
