from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model/model.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('templates/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({'prediction': str(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
