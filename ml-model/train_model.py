import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Set file paths
basedir = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(basedir, 'water_quality.csv')
MODEL_PATH = os.path.join(basedir, 'model.pkl')

# Load dataset
df = pd.read_csv(DATA_PATH)

# Clean and prepare data
df = df.dropna()  # Remove missing rows if any
df["label"] = df["label"].str.lower().map({"safe": 1, "unsafe": 0})

X = df[["color", "turbidity", "ph"]]
y = df["label"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("✅ Model trained successfully!")
print("📊 Evaluation on Test Set:")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# Save model to disk
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model saved to {MODEL_PATH}")
