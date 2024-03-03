import os
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model/mushroom_classifier.pkl')

def preprocess_data(form_data):
    df = pd.DataFrame([form_data])
    categorical_features = {
        'cap_shape': ['b', 'c', 'x', 'f', 'k', 's'],
        'cap_surface': ['f', 'g', 'y', 's'],
        'cap_color': ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
        'bruises': ['t', 'f'],
        'odor': ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],
        'gill_attachment': ['a', 'd', 'f', 'n'],
        'gill_spacing': ['c', 'w', 'd'],
        'gill_size': ['b', 'n'],
        'gill_color': ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],
        'stalk_shape': ['e', 't'],
        'stalk_root': ['b', 'c', 'u', 'e', 'z', 'r', '?'],
        'stalk_surface_above_ring': ['f', 'y', 'k', 's'],
        'stalk_surface_below_ring': ['f', 'y', 'k', 's'],
        'stalk_color_above_ring': ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
        'stalk_color_below_ring': ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
        'veil_type': ['p', 'u'],
        'veil_color': ['n', 'o', 'w', 'y'],
        'ring_number': ['n', 'o', 't'],
        'ring_type': ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'],
        'spore_print_color': ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'],
        'population': ['a', 'c', 'n', 's', 'v', 'y'],
        'habitat': ['g', 'l', 'm', 'p', 'u', 'w', 'd']
    }
    
    df_encoded = pd.DataFrame(columns=pd.get_dummies(df[categorical_features.keys()]).columns).fillna(0)

    for feature, values in categorical_features.items():
        for value in values:
            column_name = f"{feature}_{value}"
            df_encoded.at[0, column_name] = int(form_data.get(feature, '') == value)
    
    df_encoded = df_encoded.reindex(sorted(df_encoded.columns), axis=1)
    
    return df_encoded


@app.route('/', methods=['GET'])
def home():
    app.logger.info(os.getcwd())
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    data_preprocessed = preprocess_data(data)
    prediction = model.predict(data_preprocessed)
    return jsonify({'prediction': str(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
