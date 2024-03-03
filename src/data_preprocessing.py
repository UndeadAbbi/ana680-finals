import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    data = pd.read_csv(filepath, header=None)
    return data

def preprocess_data(data):
    X = data.drop(columns=[0])
    y = data[0]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_encoded = pd.get_dummies(X)
    
    return X_encoded, y_encoded

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    filepath = '../datasets/agaricus-lepiota.data'
    data = load_data(filepath)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    X_train.to_csv('../datasets/X_train.csv', index=False)
    y_train.to_csv('../datasets/y_train.csv', index=False)
    X_test.to_csv('../datasets/X_test.csv', index=False)
    y_test.to_csv('../datasets/y_test.csv', index=False)
