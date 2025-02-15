from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('crime_pattern_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# API for crime prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    year = data['year']
    month = data['month']
    day = data['day']
    hour = data['hour']
    
    # Prepare input for prediction
    input_data = np.array([[year, month, day, hour]])
    input_data_scaled = scaler.transform(input_data)
    
    # Predict crime type
    prediction = model.predict(input_data_scaled)
    
    return jsonify({'crime_type': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
