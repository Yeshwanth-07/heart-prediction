# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

# Define the prediction route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = scaler.transform([int_features])  # Scale input features and convert to 2D array

    # Make prediction
    prediction = model.predict(final_features)
    heart_conditions = {0: "No", 1: "Yes"}
    output = heart_conditions[int(prediction[0])]

    # Return result
    return render_template('index.html', prediction_text='Heart disease: {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
