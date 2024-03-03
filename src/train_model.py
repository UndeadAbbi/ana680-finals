import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_data():
    X_train = pd.read_csv('datasets/X_train.csv')
    y_train = pd.read_csv('datasets/y_train.csv', squeeze=True)
    return X_train, y_train

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename='model/model.pkl'):
    joblib.dump(model, filename)

if __name__ == "__main__":
    X_train, y_train = load_data()
    model = train_model(X_train, y_train)
    save_model(model)
