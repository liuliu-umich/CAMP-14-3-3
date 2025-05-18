from flask import Flask, render_template, request
import joblib
from features import extract_features
# from features import *
import pandas as pd
import os

app = Flask(__name__)
predict_model = joblib.load('model/model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_score', methods=['POST'])
def predict_score():
    # Get both inputs from form
    sequence = request.form['sequence']
    site = request.form['site']
    
    # Feature extraction using both inputs
    prediction_features = extract_features(sequence, site)  # Updated to accept site
    # check_non_numeric_values(prediction_features)
    print("prediction_features length: ", len(prediction_features))
    
    # Prediction
    prediction = predict_model.predict(prediction_features)[0]
    probability = predict_model.predict_proba(prediction_features)[0]

    print(prediction, probability)

    
    return render_template('result.html',
                         sequence=sequence,
                         site=site,
                         features=prediction_features,
                         prediction=prediction,
                         probability=probability)

    
    
if __name__ == '__main__':
    app.run(debug=True)