import numpy as np
import pandas as pd
import optuna
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
import matplotlib.pyplot as plt
import json

# === Load and Preprocess Data ===
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

results = []

# === Logistic Regression Optuna Objective ===
def objective_lr(trial):
    C = trial.suggest_loguniform('C', 0.001, 10)
    solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
    model = LogisticRegression(C=C, solver=solver, max_iter=300)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return f1_score(y_test, preds)

study_lr = optuna.create_study(direction='maximize')
study_lr.optimize(objective_lr, n_trials=20)
best_params_lr = study_lr.best_params
best_model_lr = LogisticRegression(**best_params_lr, max_iter=300)
best_model_lr.fit(X_train, y_train)
preds = best_model_lr.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)
results.append({
    "Model": "Logistic Regression",
    "Accuracy": acc,
    "F1-score": f1,
    "Best Params": json.dumps(best_params_lr)
})

# === Random Forest Optuna Objective ===
def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return f1_score(y_test, preds)

study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=20)
best_params_rf = study_rf.best_params
best_model_rf = RandomForestClassifier(**best_params_rf, random_state=42)
best_model_rf.fit(X_train, y_train)
preds = best_model_rf.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)
results.append({
    "Model": "Random Forest",
    "Accuracy": acc,
    "F1-score": f1,
    "Best Params": json.dumps(best_params_rf)
})

# === XGBoost Optuna Objective ===
def objective_xgb(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
    model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                          learning_rate=learning_rate, use_label_encoder=False,
                          eval_metric='logloss')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return f1_score(y_test, preds)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=20)
best_params_xgb = study_xgb.best_params
best_model_xgb = XGBClassifier(**best_params_xgb, use_label_encoder=False, eval_metric='logloss')
best_model_xgb.fit(X_train, y_train)
preds = best_model_xgb.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)
results.append({
    "Model": "XGBoost",
    "Accuracy": acc,
    "F1-score": f1,
    "Best Params": json.dumps(best_params_xgb)
})

# === CHARLIE Optuna Objective ===
def objective_charlie(trial):
    selected_features = trial.suggest_int('selected_features', 4, 8)
    hidden_layer_1 = trial.suggest_int('hl1', 32, 256)
    hidden_layer_2 = trial.suggest_int('hl2', 16, 128)
    hidden_layer_3 = trial.suggest_int('hl3', 8, 64)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    epochs = trial.suggest_int('epochs', 20, 60)
    
    model = CHARLIE(
        input_dim=X_train.shape[1],
        selected_features=selected_features,
        rf_trees=100,
        hidden_layers=(hidden_layer_1, hidden_layer_2, hidden_layer_3),
        classification=True
    )
    model.train_model(X_train, y_train, epochs=epochs, lr=lr)
    preds = model.predict(X_test)
    preds_binary = np.argmax(preds, axis=1)
    return f1_score(y_test, preds_binary)

study_charlie = optuna.create_study(direction='maximize')
study_charlie.optimize(objective_charlie, n_trials=20)
best_params_charlie = study_charlie.best_params
# Train best CHARLIE model
charlie = CHARLIE(
    input_dim=X_train.shape[1],
    selected_features=best_params_charlie['selected_features'],
    rf_trees=100,
    hidden_layers=(
        best_params_charlie['hl1'],
        best_params_charlie['hl2'],
        best_params_charlie['hl3']
    ),
    classification=True
)
charlie.train_model(X_train, y_train, epochs=best_params_charlie['epochs'], lr=best_params_charlie['lr'])
preds = charlie.predict(X_test)
preds_binary = np.argmax(preds, axis=1)
acc = accuracy_score(y_test, preds_binary)
f1 = f1_score(y_test, preds_binary)
results.append({
    "Model": "CHARLIE",
    "Accuracy": acc,
    "F1-score": f1,
    "Best Params": json.dumps(best_params_charlie)
})

# === Combine Results ===
results_df = pd.DataFrame(results)
print("\n=== Final Results ===")
print(results_df)

# === Visualization ===
plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['Accuracy'], alpha=0.6, label='Accuracy')
plt.plot(results_df['Model'], results_df['F1-score'], color='red', marker='o', label='F1-score')
plt.title('Optuna Tuned Model Performance')
plt.xlabel('Model')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
