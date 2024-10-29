from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


model = pickle.load(open('diabetes_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json
    features = [data['Pregnancies'], data['Glucose'], data['BloodPressure'],
                data['SkinThickness'], data['Insulin'], data['BMI'],
                data['DiabetesPedigreeFunction'], data['Age']]

    features = np.array(features).reshape(1, -1)
    
    prediction = model.predict(features)

    return jsonify({'Outcome': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
