# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# %%
data = pd.read_csv('datasets/agaricus-lepiota.data', header=None)

# %%
print(data.head())
print(data.isnull().sum())

# %%
target_counts = data[0].value_counts()
plt.figure(figsize=(6,4))
target_counts.plot(kind='bar')
plt.title('Distribution of Target Variable')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

# %%
le = LabelEncoder()
y_encoded = le.fit_transform(data[0])

X_encoded = pd.get_dummies(data.drop(columns=[0]))

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# %%

print(f"Accuracy: {accuracy_score(y_test, predictions)}")
print(f"Precision: {precision_score(y_test, predictions)}")
print(f"Recall: {recall_score(y_test, predictions)}")
print(f"F1 Score: {f1_score(y_test, predictions)}")

# %%
joblib.dump(model, 'model/mushroom_classifier.pkl')
print("Model saved.")



