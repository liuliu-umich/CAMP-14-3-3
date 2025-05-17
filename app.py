from flask import Flask, render_template, request
import joblib
from features import extract_features

app = Flask(__name__)
# model = joblib.load('model/model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get both inputs from form
    sequence = request.form['sequence']
    site = request.form['site']
    
    # Feature extraction using both inputs
    features = extract_features(sequence, site)  # Updated to accept site
    
    # Prediction
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0]
    
    return render_template('result.html',
                         sequence=sequence,
                         site=site,
                         features=features,
                         prediction=prediction,
                         probability=probability)
    
    
if __name__ == '__main__':
    app.run(debug=True)