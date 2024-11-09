import json
import boto3
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Initialize S3 client and specify model location
s3 = boto3.client('s3')
bucket_name = 'mymodelaws'
model_key = 'modelaws.joblib'
download_path = '/tmp/modelaws.joblib'  # Temporary storage for model

# SNS Client
sns_client = boto3.client('sns', region_name='us-east-1')
sns_topic_arn = 'arn:aws:sns:us-east-1:009160066859:diabetesnotif'

# DynamoDB Client
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table_name = 'DiabetesPrediction'  # DynamoDB table name to store results
table = dynamodb.Table(table_name)

# Download and load the model when the server starts
try:
    s3.download_file(bucket_name, model_key, download_path)
    model = joblib.load(download_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is live'})

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model could not be loaded'}), 500

    try:
        # Get JSON data from the request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid or missing JSON data'}), 400

        email = data.get('email')
        if not email:
            return jsonify({'error': 'Email is required'}), 400

        # Extract the rest of the input data
        pregnancies = int(data.get('Pregnancies', 0))
        glucose = int(data.get('Glucose', 0))
        blood_pressure = int(data.get('BloodPressure', 0))
        skin_thickness = int(data.get('SkinThickness', 0))
        insulin = int(data.get('Insulin', 0))
        bmi = float(data.get('BMI', 0.0))
        pedigree = float(data.get('DiabetesPedigreeFunction', 0.0))
        age = int(data.get('Age', 0))

        # Check if email is already subscribed
        response = sns_client.list_subscriptions_by_topic(TopicArn=sns_topic_arn)
        subscriptions = response.get('Subscriptions', [])
        if not any(sub['Endpoint'] == email for sub in subscriptions):
            sns_client.subscribe(
                TopicArn=sns_topic_arn,
                Protocol='email',
                Endpoint=email
            )

        # Prepare input data for prediction
        input_data = [[pregnancies, glucose, blood_pressure, skin_thickness,
                       insulin, bmi, pedigree, age]]
        input_df = pd.DataFrame(input_data, columns=[
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ])

        # Make prediction
        prediction = model.predict(input_df)
        predicted_label = int(prediction[0])  # 0 or 1

        # Send SNS notification
        message = f"Diabetes Prediction Result: {predicted_label}"
        sns_client.publish(
            TopicArn=sns_topic_arn,
            Message=message,
            Subject='Diabetes Prediction Result'
        )

        # Store the prediction result in DynamoDB
        result_data = {
            'email': email,
            'timestamp': datetime.utcnow().isoformat(),
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': pedigree,
            'Age': age,
            'PredictedOutcome': predicted_label
        }

        # Save to DynamoDB
        table.put_item(Item=result_data)

        return jsonify({'Predicted Label': predicted_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
