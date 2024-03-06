import os
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model/mushroom_classifier.pkl')

def preprocess_data(form_data):
    df_encoded = pd.DataFrame(columns=model.feature_names_in_)
    for column in df_encoded.columns:
        feature, value = column.rsplit('_', 1)
        form_value = form_data.get(feature, None)
        if form_value == value:
            df_encoded.at[0, column] = 1
        else:
            df_encoded.at[0, column] = 0
    return df_encoded

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    data_preprocessed = preprocess_data(data)
    prediction = model.predict(data_preprocessed)
    result = "edible" if prediction[0] == 0 else "poisonous"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
