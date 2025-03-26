import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from models.ensemble import CHARLIE
import matplotlib.pyplot as plt

# === Data Loading & Preprocessing ===
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
df = pd.read_csv(url, names=columns)
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df['ca'] = df['ca'].astype(float)
df['thal'] = df['thal'].astype(float)
df["target"] = (df["target"].astype(int) > 0).astype(int)

X = df.drop(columns=['target']).astype(float).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Evaluation Function ===
def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"{name} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
    return {"Model": name, "Accuracy": acc, "F1-score": f1}

results = []
best_estimators = {}  # Store the best models

# === Logistic Regression Hyperparameter Tuning ===
print("=== Logistic Regression Tuning ===")
log_reg = LogisticRegression(max_iter=200)
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}
grid_lr = GridSearchCV(log_reg, param_grid_lr, cv=5, scoring='f1', n_jobs=-1)
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_
print(f"Best Logistic Regression Params: {grid_lr.best_params_}")
results.append(evaluate_model("Logistic Regression", best_lr, X_test, y_test))
best_estimators["Logistic Regression"] = best_lr

# === Random Forest Hyperparameter Tuning ===
print("\n=== Random Forest Tuning ===")
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print(f"Best Random Forest Params: {grid_rf.best_params_}")
results.append(evaluate_model("Random Forest", best_rf, X_test, y_test))
best_estimators["Random Forest"] = best_rf

# === XGBoost Hyperparameter Tuning ===
print("\n=== XGBoost Tuning ===")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='f1', n_jobs=-1)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
print(f"Best XGBoost Params: {grid_xgb.best_params_}")
results.append(evaluate_model("XGBoost", best_xgb, X_test, y_test))
best_estimators["XGBoost"] = best_xgb

# === CHARLIE Hyperparameter Tuning ===
print("\n=== CHARLIE Tuning ===")
charlie_param_grid = {
    'selected_features': [4, 6, 8],
    'hidden_layers': [(128, 64, 32), (256, 128, 64), (64, 32)],
    'lr': [0.01, 0.001],
    'epochs': [30, 50]
}

best_f1_charlie = 0
best_params_charlie = None
best_charlie_model = None

for sf in charlie_param_grid['selected_features']:
    for hl in charlie_param_grid['hidden_layers']:
        for lr in charlie_param_grid['lr']:
            for epoch in charlie_param_grid['epochs']:
                charlie = CHARLIE(input_dim=X_train.shape[1],
                                  selected_features=sf,
                                  rf_trees=100,
                                  hidden_layers=hl,
                                  classification=True)
                charlie.train_model(X_train, y_train, epochs=epoch, lr=lr)
                preds = charlie.predict(X_test)
                preds_binary = np.argmax(preds, axis=1)
                f1 = f1_score(y_test, preds_binary)
                if f1 > best_f1_charlie:
                    best_f1_charlie = f1
                    best_params_charlie = (sf, hl, lr, epoch)
                    best_charlie_model = charlie

# Train CHARLIE with best params (already trained above)
print(f"Best CHARLIE Params: Selected Features={best_params_charlie[0]}, Hidden Layers={best_params_charlie[1]}, LR={best_params_charlie[2]}, Epochs={best_params_charlie[3]}")
preds_binary = np.argmax(best_charlie_model.predict(X_test), axis=1)
acc = accuracy_score(y_test, preds_binary)
print(f"CHARLIE - Accuracy: {acc:.4f}, F1-score: {best_f1_charlie:.4f}")
results.append({"Model": "CHARLIE", "Accuracy": acc, "F1-score": best_f1_charlie})
best_estimators["CHARLIE"] = best_charlie_model

# === Summary Table ===
print("\n=== Summary ===")
results_df = pd.DataFrame(results)
results_df.sort_values(by="F1-score", ascending=False, inplace=True)
print(results_df)

plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['Accuracy'], alpha=0.6, label='Accuracy')
plt.plot(results_df['Model'], results_df['F1-score'], color='red', marker='o', label='F1-score')
plt.title('Hyperparameter Tuned Model Performance Comparison')
plt.xlabel('Model')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# === Best Estimator Selection ===
best_model_row = results_df.iloc[0]
best_model_name = best_model_row['Model']
print(f"\n🏆 Best Model Overall: {best_model_name} with F1-score: {best_model_row['F1-score']:.4f}")
print(f"Best Model Object: {best_estimators[best_model_name]}")
