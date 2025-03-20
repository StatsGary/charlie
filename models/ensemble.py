import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class CHARLIE(nn.Module):
    def __init__(
        self,
        input_dim,
        selected_features=5,
        rf_trees=100,
        alpha=0.5,
        hidden_layers=(32, 16),
        classification=False,
        random_state=None,
        device=None,
        logging_enabled=True,
        checkpoint_dir=None,
        **kwargs
    ):
        super(CHARLIE, self).__init__()

        # Device config
        self.device = torch.device("cuda" if torch.cuda.is_available() and device is None else "cpu")
        self.logging_enabled = logging_enabled

        # Random Forest setup
        rf_params = {"n_estimators": rf_trees, **kwargs}
        if random_state is not None:
            rf_params["random_state"] = random_state

        if classification:
            self.rf = RandomForestClassifier(**rf_params)
        else:
            self.rf = RandomForestRegressor(**rf_params)

        self.classification = classification
        self.selected_features = min(selected_features, input_dim)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.top_features = None
        self.nn_model = None

        if logging_enabled:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            logging.info(f"Using device: {self.device}")

        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _build_nn(self):
        layers = []
        layer_dims = [self.selected_features] + list(self.hidden_layers)

        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(nn.ReLU())

        output_dim = self.num_classes if self.classification else 1
        layers.append(nn.Linear(layer_dims[-1], output_dim))

        if self.classification:
            layers.append(nn.Softmax(dim=1))

        self.nn_model = nn.Sequential(*layers).to(self.device)

        if self.logging_enabled:
            logging.info("Neural Network built successfully.")

    def forward(self, x):
        if self.nn_model is None or self.top_features is None:
            raise ValueError("Model has not been fully trained yet.")

        x = x.contiguous()

        # RF prediction on full feature set
        x_cpu = x.cpu().numpy()
        if self.classification:
            rf_preds = self.rf.predict_proba(x_cpu)
        else:
            rf_preds = self.rf.predict(x_cpu).reshape(-1, 1) 

        rf_tensor = torch.tensor(rf_preds, dtype=torch.float32, device=self.device)

        # Slice selected features for NN
        top_features_tensor = torch.tensor(self.top_features, dtype=torch.long, device=self.device)
        x_selected = torch.index_select(x, 1, top_features_tensor)

        ann_preds = self.nn_model(x_selected)

        return self.alpha * rf_tensor + (1 - self.alpha) * ann_preds

    def train_model(self, train_features, train_targets, epochs=50, lr=0.001):
        train_features = np.ascontiguousarray(train_features.copy())
        self.rf.fit(train_features, train_targets)
        feature_importance = self.rf.feature_importances_
        self.top_features = np.argsort(feature_importance)[::-1][:self.selected_features].copy()
        if len(self.top_features) == 0:
            raise ValueError("No valid features selected by Random Forest.")

        if self.classification:
            self.num_classes = len(np.unique(train_targets))
        else:
            self.num_classes = 1
        self._build_nn()

        criterion = nn.CrossEntropyLoss() if self.classification else nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        X_train_tensor = torch.tensor(train_features, dtype=torch.float32, device=self.device)
        if self.classification:
            y_train_tensor = torch.tensor(train_targets, dtype=torch.long, device=self.device)
        else:
            y_train_tensor = torch.tensor(train_targets, dtype=torch.float32, device=self.device).unsqueeze(1)

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            preds = self.forward(X_train_tensor)
            loss = criterion(preds, y_train_tensor)

            loss.backward()
            optimizer.step()

            if self.logging_enabled:
                logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

            if self.checkpoint_dir is not None:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }, checkpoint_path)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = np.ascontiguousarray(x.copy())
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
            else:
                x = x.to(self.device).contiguous()

            combined_output = self.forward(x)

        return combined_output.cpu().numpy()
