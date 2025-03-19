import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from models.ensemble import CHARLIE

# Load and preprocess data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
df = pd.read_csv(url, names=columns)
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df['ca'] = df['ca'].astype(float)
df['thal'] = df['thal'].astype(float)
df["target"] = (df["target"].astype(int) > 0).astype(int)

X = df.drop(columns=['target']).astype(float).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Evaluation function
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"{name} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
    return {"Model": name, "Accuracy": acc, "F1-score": f1}

results = []
# Traditional Models
print("=== Traditional Models ===")
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False)
}

for name, model in models.items():
    res = evaluate_model(name, model, X_train, y_train, X_test, y_test)
    results.append(res)

# CHARLIE Model
print("\n=== CHARLIE Model ===")
charlie = CHARLIE(
    input_dim=X_train.shape[1],
    selected_features=6, 
    rf_trees=100,
    hidden_layers=(128, 64, 32),
    classification=True
)
charlie.train_model(X_train, y_train, epochs=50, lr=0.001)
charlie_preds = charlie.predict(X_test)
charlie_preds_binary = np.argmax(charlie_preds, axis=1)

acc = accuracy_score(y_test, charlie_preds_binary)
f1 = f1_score(y_test, charlie_preds_binary)
print(f"CHARLIE - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
results.append({"Model": "CHARLIE", "Accuracy": acc, "F1-score": f1})

# Summary Table
print("\n=== Summary ===")
results_df = pd.DataFrame(results)
results_df.sort_values(by="F1-score", 
                       ascending=False).to_string(index=False)

print(results_df)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], 
        results_df['Accuracy'], 
        alpha=0.6, label='Accuracy')
plt.plot(results_df['Model'], 
         results_df['F1-score'], 
         color='red', 
         marker='o', 
         label='F1-score')
plt.title('Model Performance Comparison')
plt.xlabel('Model')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()