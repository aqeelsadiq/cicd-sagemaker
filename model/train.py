# model/train.py

import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# -----------------------------
# 1️⃣ Load real dataset (CSV)
# -----------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

df = pd.read_csv(url, names=columns)

# Encode species to numeric labels
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Features and labels
X = df.drop("species", axis=1).values
y = df['species'].values

# Shuffle dataset
X, y = shuffle(X, y, random_state=42)

print(f"Dataset size: {X.shape[0]} samples")

# -----------------------------
# 2️⃣ Split dataset (optional)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 3️⃣ Train model
# -----------------------------
model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Test accuracy
acc = model.score(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# -----------------------------
# 4️⃣ Save model
# -----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")



#this below is working and tested
# from sklearn.linear_model import LogisticRegression
# from sklearn.datasets import load_iris
# import joblib
# import os

# # Load Iris dataset
# X, y = load_iris(return_X_y=True)

# # Train Logistic Regression model
# model = LogisticRegression(max_iter=200)
# model.fit(X, y)

# # Save model using joblib
# model_dir = os.path.dirname(__file__)  # Save in model folder
# joblib.dump(model, os.path.join(model_dir, "model.pkl"))

# print("Model trained and saved as model.pkl")
