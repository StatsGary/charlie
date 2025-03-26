# CHARLIE (Combined Alpha-weighted Random Forest Layered Inference Ensemble)

![](https://github.com/StatsGary/charlie/blob/29edeedc18b0f63bef2c427d982bd0ffdd1691f4/fig/CHARLIE_logo.png)

![GitHub Actions](https://github.com/StatsGary/charlie/actions/workflows/python-package.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/charlie.svg)](https://pypi.org/project/charlie/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

CHARLIE is an acronym that encapsulates the core process of this model. Standing for:

- Combined: blending two modeling techniques (Random Forest & Neural Networks)
- Alpha-weighted: the learnable parameter that controls the blending $a$
- Random Forest: used for feature extraction
- Layered: the structure of the neural network contains multiple layers
- Inference Ensemble: Final predictive ensemble combining RF and NN outputs.

Why it is really called CHARLIE? I am sure only my son knows that ❤️.


## Importing CHARLIE to perform ensembling

To import the package we go to the below:

```bash
pip install charliepy
```
This will get the project from PyPi: <some url> and then you can import the model using:

```python
from charlie.models.ensemble import CHARLIE
```

## Overview

The CHARLIE class implements a hybrid ML model that combines: 
- **Random Forest (RF)** for feature importance ranking and initial predictions
- **Feedforward Neural Network (NN)** for learning non-linear relationships on selected top features
- **Learnable weighting parameter** that blends predictions from both models


## Model architecture

Consists of two models: 

- Random Forest trained on the entire feature set and outputs either class probs or continuous predictions. 
- Neural Network - built after using a reduced features set based on RF feature importance


## Training Process

1. **Random Forest Training**:

    * Trained on full feature set (all our $X$ features)
    * Outputs the importance $I$ of each feature i.e. how much each feature affects the prediction
2. Feature Selection:

    * Select top `selected_features` based on their importance $I$
3. Neural Network Building:
    * NN input dimension is those `selected features`
    * These are configured according to the number of `hidden_layers` passed as a Tuple to the Neural Network
4. Neural Network Training:

* Loss Function:
    
    - **Classification**: Cross Entropy Loss (https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

    - **Regression**: Mean Squared Error Loss (https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)

* Optimiser: ADAM (https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)

* Training updates both:

    - NN weights $\theta{}_{NN}$ 
    - Blending parameter $\alpha$


## Mathematical Formulation Summary

$$\hat{\mathbf{y}} = \alpha\cdot f_\text{RF}(\mathbf{X})+(1-\alpha) \cdot f_\text{NN}(\mathbf{X}_\text{top})$$

where:
- $\alpha$ is trained alongside $\text{NN}$ parameters
- $f_\text{RF}$ is trained first

## How to use CHARLIE? 

The first step, we will gather the imports that we need:

```python
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
from charliePy.models.ensemble import CHARLIE
```
### Preprocess data

The next stage is to preprocess the heart disease classification data we are going to need to use:

```python
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
```

### Split and scale 

We will now split the data ino training and testing splits, ready to be used: 
```python
# Split our data into train and test splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Evaluation step

In this step, we will create an evaluation function for the project:

```python
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """
    Function to use accuracy and F1 score as our measures
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"{name} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
    return {"Model": name, "Accuracy": acc, "F1-score": f1}
```

## Modelling with our baseline models

We will use a Logistic Regression, Random Forest and Boosted Forest (XGBoost) to prepare our comparisons:

```python
results = []
print("=== Traditional Models ===")
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False)
}

for name, model in models.items():
    res = evaluate_model(name, model, X_train, y_train, X_test, y_test)
    results.append(res)
```

The loop at the end iterates through the model versions and finds appends the evaluated model results to the empty list. 

### Using CHARLIE

In this step, we will use CHARLIE to do the training:

```python
charlie = CHARLIE(
    input_dim=X_train.shape[1],
    selected_features=6, 
    rf_trees=100,
    hidden_layers=(128, 64, 32),
    classification=True
)
charlie.train_model(X_train, y_train, epochs=50, lr=0.001)
```

The model will train, do the feature selection and then train the network, as outlined in the training section above. 

Once trained, we can use the instantiated class to reveal the predict class method, this will be useful for using against our test set:

```python
charlie_preds = charlie.predict(X_test)
charlie_preds_binary = np.argmax(charlie_preds, axis=1
```
Now we have the predictions, we will use the same metrics and append our results from the CHARLIE model and then do a model comparison:

```python
acc = accuracy_score(y_test, charlie_preds_binary)
f1 = f1_score(y_test, charlie_preds_binary)
print(f"CHARLIE - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
results.append({"Model": "CHARLIE", "Accuracy": acc, "F1-score": f1})

# Store results in DataFrame
results_df = pd.DataFrame(results)
results_df.sort_values(
    by="F1-score", 
    ascending=False).to_string(index=False)
```

### Compare CHARLIE to baseline models

The following visualisation will compare the CHARLIE model to the baseline models we chose:

```python
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
```

This produces the visualisation illustrated below:

![](https://github.com/StatsGary/charlie/blob/29edeedc18b0f63bef2c427d982bd0ffdd1691f4/fig/CHARLIE.png)

Due to combining our feature selector with a neural network, we can beat the standard Random Forest classifier on its own, as well as XGBoost, which shows the power of this approach, as `accuracy=0.9` and `F1-Score=0.869`.

## Performance on regression credit modelling 

This was contributed by https://www.linkedin.com/in/venkateshwari-narayanan-668661176/. 

🚀 Credit Score Prediction Framework: We recently tested the CHARLIE Model (Combined Alpha-weighted Random Forest Layered Inference Ensemble) on a credit score prediction task and compared its performance against several traditional models. The results were remarkable, demonstrating CHARLIE’s ability to balance accuracy and robustness.

### 📈 Visualizing Performance Metrics

To further highlight CHARLIE's performance, the following visual comparisons showcase results for key metrics across models chosen for regression:

![](fig/credit_risk_regression.jpg)

### ⚡️ Key Insights:

- 🎯 The CHARLIE Model (Linear) matched the performance of Linear Regression with near-perfect accuracy (R² = 0.9999) while maintaining low error values across MSE, RMSE, and MAE.

- 🎯It demonstrated high stability in cross-validation, with a mean cross-validated MSE of 0.0804 and a minimal standard deviation of 0.0021.

- 🎯 Compared to traditional models like Ridge, Lasso, and Decision Trees, CHARLIE exhibited superior generalization and significantly lower variance.

- 🎯Gradient Boosting also showed strong performance but was slightly outperformed by the CHARLIE Model.

### 🎯 Potential Applications:
The CHARLIE Model’s exceptional performance makes it well-suited for:

- Financial risk analysis and credit score prediction.

- Regression tasks involving structured, tabular data with complex relationships.

- Datasets where high accuracy and consistency are critical.

