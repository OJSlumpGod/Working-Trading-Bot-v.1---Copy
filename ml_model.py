import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tsfresh import extract_features
from skorch import NeuralNetClassifier
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import joblib
import logging
import talib
import json

# Import configuration parameters
from config import PYTORCH_HIDDEN_SIZE, PYTORCH_LEARNING_RATE, PYTORCH_EPOCHS, MODEL_CONFIG

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ForexNet(nn.Module):
    """
    Neural network architecture for Forex prediction using PyTorch.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(ForexNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out  # Corrected from 'return x' to 'return out'


class MLModel:
    """
    MLModel handles feature engineering, model training, prediction, and model persistence.
    It utilizes an ensemble of classical ML models and a PyTorch neural network for predictions.
    """

    def __init__(self):
        # Initialize logger for MLModel
        self.logger = logging.getLogger('ml_model')
        self.logger.setLevel(logging.INFO)

        # File handler for MLModel logs
        ml_file_handler = logging.FileHandler(os.path.join('logs', 'ml_model.log'))
        ml_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ml_file_handler.setFormatter(ml_formatter)
        self.logger.addHandler(ml_file_handler)

        # Console handler for MLModel logs
        ml_console_handler = logging.StreamHandler()
        ml_console_handler.setFormatter(ml_formatter)
        self.logger.addHandler(ml_console_handler)

        self.logger.info("[MLModel] Initializing model pipeline...")

        # Preprocessors
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=7)

        # Model parameters from config
        self.pytorch_hidden_size = PYTORCH_HIDDEN_SIZE
        self.pytorch_epochs = PYTORCH_EPOCHS
        self.pytorch_lr = PYTORCH_LEARNING_RATE

        # Model directories
        self.models_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.logger.info(f"[MLModel] Models directory: {self.models_dir}")

        self.model_files = {
            "sgd": os.path.join(self.models_dir, "sgd_model.pkl"),
            "rf": os.path.join(self.models_dir, "rf_model.pkl"),
            "gb": os.path.join(self.models_dir, "gb_model.pkl"),
            "voting": os.path.join(self.models_dir, "voting_model.pkl"),
            "pytorch": os.path.join(self.models_dir, "pytorch_model.pth"),
            "scaler": os.path.join(self.models_dir, "scaler.pkl"),
            "pca": os.path.join(self.models_dir, "pca.pkl")
        }

        # Initialize or load classical models
        self.models = {
            "sgd": self.load_model("sgd", SGDClassifier(max_iter=1000, tol=1e-3)),
            "rf": self.load_model("rf", RandomForestClassifier(n_estimators=100)),
            "gb": self.load_model("gb", GradientBoostingClassifier(n_estimators=100))
        }

        # Initialize Voting Classifier
        self.voting_model = VotingClassifier(
            estimators=[('sgd', self.models['sgd']),
                       ('rf', self.models['rf']),
                       ('gb', self.models['gb'])],
            voting='hard'
        )
        self.logger.info("[MLModel] Voting classifier initialized.")

        # Load Voting Classifier if it exists
        if os.path.exists(self.model_files["voting"]):
            self.logger.info("[MLModel] Loading existing Voting classifier.")
            try:
                self.voting_model = joblib.load(self.model_files["voting"])
                self.logger.info("[MLModel] Voting classifier loaded successfully.")
            except Exception as e:
                self.logger.error(f"[MLModel] Failed to load Voting classifier: {e}")
        else:
            self.logger.info("[MLModel] Voting classifier not trained yet. It will be trained during model training.")

        # Initialize or load PyTorch model
        self.pytorch_model = self.build_pytorch_model()
        if os.path.exists(self.model_files["pytorch"]):
            self.logger.info("[MLModel] Loading existing PyTorch model state.")
            try:
                self.pytorch_model.load_state_dict(torch.load(self.model_files["pytorch"], map_location=device))
                self.pytorch_model.to(device)
                self.logger.info("[MLModel] PyTorch model loaded successfully.")
            except RuntimeError as e:
                self.logger.error(f"[MLModel] RuntimeError while loading PyTorch model: {e}")
            except Exception as e:
                self.logger.error(f"[MLModel] Unexpected error while loading PyTorch model: {e}")
        else:
            self.logger.info("[MLModel] No existing PyTorch model found. Initializing a new one.")

        # Load Scaler and PCA
        self.load_scaler_pca()

        # Stats
        self.successful_trades = 0
        self.failed_trades = 0

    def is_ready(self):
        """
        Check if scaler and PCA are fitted.
        """
        return hasattr(self.scaler, 'scale_') and hasattr(self.pca, 'components_')

    def load_scaler_pca(self):
        """
        Load the scaler and PCA objects from disk if they exist.
        """
        try:
            scaler_path = self.model_files["scaler"]
            pca_path = self.model_files["pca"]
            
            if os.path.exists(scaler_path) and os.path.exists(pca_path):
                self.scaler = joblib.load(scaler_path)
                self.pca = joblib.load(pca_path)
                self.logger.info("[MLModel] Scaler and PCA loaded successfully.")
            else:
                self.logger.warning("[MLModel] Scaler and/or PCA files not found. They need to be fitted during training.")
        except Exception as e:
            self.logger.error(f"[MLModel] Failed to load Scaler and PCA - {e}")

    def save_scaler_pca(self):
        """
        Save the scaler and PCA objects to disk.
        """
        try:
            joblib.dump(self.scaler, self.model_files["scaler"])
            self.logger.info(f"[MLModel] Scaler saved at {self.model_files['scaler']}")
            
            joblib.dump(self.pca, self.model_files["pca"])
            self.logger.info(f"[MLModel] PCA saved at {self.model_files['pca']}")
        except Exception as e:
            self.logger.error(f"[MLModel] Failed to save Scaler and PCA - {e}")

    def load_model(self, model_name, default_model=None):
        """
        Load a specific model from disk or return a default model.
        """
        filepath = self.model_files[model_name]
        if os.path.exists(filepath):
            try:
                self.logger.info(f"[MLModel] Loading {model_name} from {filepath}")
                return joblib.load(filepath)
            except Exception as e:
                self.logger.error(f"[MLModel] Failed to load {model_name} from {filepath} - {e}")
                return default_model
        else:
            self.logger.info(f"[MLModel] {model_name} not found. Using default model.")
            return default_model

    def save_model(self, model_name, model):
        """
        Save a model to the specified filepath.
        """
        filepath = self.model_files[model_name]
        try:
            joblib.dump(model, filepath)
            self.logger.info(f"[MLModel] {model_name} model saved at {filepath}")
        except Exception as e:
            self.logger.error(f"[MLModel] Failed to save {model_name} model at {filepath} - {e}")

    def build_pytorch_model(self):
        """
        Build the neural network architecture for PyTorch.
        """
        input_size = 7
        hidden_size = self.pytorch_hidden_size
        output_size = 1

        model = ForexNet(input_size, hidden_size, output_size)
        return model

    def save_pytorch_model(self):
        """
        Save the PyTorch model's state dictionary.
        """
        try:
            torch.save(self.pytorch_model.state_dict(), self.model_files["pytorch"])
            self.logger.info("[MLModel] PyTorch model saved successfully.")
        except Exception as e:
            self.logger.error(f"[MLModel] Failed to save PyTorch model - {e}")

    def prepare_features(self, price_data, training=False):
        """
        Prepare features from OANDA price data for model input,
        including various technical indicators.
        
        Args:
            price_data (dict): Price data fetched from OANDA.
            training (bool): Flag indicating if the method is called during training.
        
        Returns:
            np.ndarray: Preprocessed feature array.
        """
        self.logger.info("[MLModel] Starting feature engineering.")
        try:
            closes = np.array([float(cd['mid']['c']) for cd in price_data['candles']])
            highs = np.array([float(cd['mid']['h']) for cd in price_data['candles']])
            lows = np.array([float(cd['mid']['l']) for cd in price_data['candles']])
            volumes = np.array([float(cd['volume']) for cd in price_data['candles']])

            # Calculate technical indicators
            sma_10 = talib.SMA(closes, timeperiod=10)
            sma_50 = talib.SMA(closes, timeperiod=50)
            rsi = talib.RSI(closes, timeperiod=14)
            upper_band, mid_band, lower_band = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2)
            atr = talib.ATR(highs, lows, closes, timeperiod=14)
            macd, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
            slowk, slowd = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowd_period=3)
            cci = talib.CCI(highs, lows, closes, timeperiod=14)
            willr = talib.WILLR(highs, lows, closes, timeperiod=14)
            adx = talib.ADX(highs, lows, closes, timeperiod=14)
            vwap = np.cumsum(volumes * closes) / np.cumsum(volumes)

            # Stack features into a single array
            feats = np.column_stack((
                sma_10, sma_50, rsi, upper_band, lower_band, atr, macd_hist,
                slowk, slowd, cci, willr, adx, vwap, macd, macd_signal
            ))

            # Handle NaN values by removing rows with any NaNs
            valid_indices = ~np.isnan(feats).any(axis=1)
            feats = feats[valid_indices]
            self.logger.info(f"[MLModel] Processed {feats.shape[0]} valid rows of features.")

            if training:
                # Fit PCA first
                pca_feats = self.pca.fit_transform(feats)
                self.logger.info("[MLModel] PCA transformation fitted successfully.")
                # Then fit scaler on PCA-transformed features
                self.scaler.fit(pca_feats)
                self.logger.info("[MLModel] Scaler fitted successfully.")
                # Transform PCA features using the fitted scaler
                scaled = self.scaler.transform(pca_feats)
                self.logger.info("[MLModel] Feature scaling applied successfully.")
            else:
                if not self.is_ready():
                    self.logger.error("[MLModel] Scaler or PCA not fitted. Cannot transform features.")
                    return np.empty((0, 7))
                # Apply only transform
                pca_feats = self.pca.transform(feats)
                self.logger.info("[MLModel] PCA transformation applied successfully.")
                scaled = self.scaler.transform(pca_feats)
                self.logger.info("[MLModel] Feature scaling applied successfully.")

            return scaled
        except Exception as e:
            self.logger.error(f"[MLModel] Error during feature engineering - {e}")
            return np.empty((0, 7))

    def train_pytorch_model(self, X_train, y_train, X_val, y_val):
        """
        Train the PyTorch neural network with early stopping and validation monitoring.
        """
        self.logger.info("[MLModel] Starting PyTorch model training.")
        patience = 5
        best_val_loss = float('inf')
        patience_counter = 0

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.pytorch_model.parameters(), lr=self.pytorch_lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)

        # Convert data to PyTorch tensors
        xt = torch.tensor(X_train, dtype=torch.float32).to(device)
        yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        xv = torch.tensor(X_val, dtype=torch.float32).to(device)
        yv = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

        self.pytorch_model.to(device)
        self.pytorch_model.train()

        for epoch in range(self.pytorch_epochs):
            optimizer.zero_grad()
            output = self.pytorch_model(xt)
            loss = criterion(output, yt)
            loss.backward()
            optimizer.step()

            # Validation phase
            self.pytorch_model.eval()
            with torch.no_grad():
                val_output = self.pytorch_model(xv)
                val_loss = criterion(val_output, yv)

            self.logger.info(f"[MLModel] Epoch {epoch+1}: Train Loss={loss.item():.6f}, Validation Loss={val_loss.item():.6f}")

            # Early stopping logic
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"[MLModel] Early stopping triggered at epoch {epoch+1}.")
                    break

            self.pytorch_model.train()
            scheduler.step(val_loss)

    def cross_validate_pytorch_model(self, X_train, y_train):
        """
        Perform cross-validation on the PyTorch model using Skorch.
        """
        self.logger.info("[MLModel] Starting cross-validation for PyTorch model.")
        try:
            net = NeuralNetClassifier(
                module=ForexNet,
                module__input_size=7,
                module__hidden_size=self.pytorch_hidden_size,
                module__output_size=1,
                max_epochs=self.pytorch_epochs,
                lr=self.pytorch_lr,
                device=device,
                iterator_train__shuffle=True,
                callbacks=[('lr_scheduler', ReduceLROnPlateau(optimizer=optim.Adam, mode='min'))]
            )
            param_grid = {"lr": [0.001, 0.01], "max_epochs": [50, 100]}
            gs = GridSearchCV(net, param_grid, cv=3, scoring='accuracy', verbose=2)
            gs.fit(X_train, y_train)
            self.logger.info(f"[MLModel] Best parameters for PyTorch model: {gs.best_params_}")
        except Exception as e:
            self.logger.error(f"[MLModel] Cross-validation failed - {e}")

    def train_classical_models(self, X_train, y_train):
        """
        Train classical machine learning models: SGD, Random Forest, and Gradient Boosting.
        """
        self.logger.info("[MLModel] Starting training for classical ML models.")
        try:
            # Train SGDClassifier
            self.logger.info("[MLModel] Training SGDClassifier.")
            self.models["sgd"].fit(X_train, y_train)
            self.save_model("sgd", self.models["sgd"])

            # Train RandomForestClassifier
            self.logger.info("[MLModel] Training RandomForestClassifier.")
            self.models["rf"].fit(X_train, y_train)
            self.save_model("rf", self.models["rf"])

            # Train GradientBoostingClassifier
            self.logger.info("[MLModel] Training GradientBoostingClassifier.")
            self.models["gb"].fit(X_train, y_train)
            self.save_model("gb", self.models["gb"])

            # Train VotingClassifier
            self.logger.info("[MLModel] Training VotingClassifier.")
            self.voting_model.fit(X_train, y_train)
            self.save_model("voting", self.voting_model)

            self.logger.info("[MLModel] Classical ML models trained and saved successfully.")
        except Exception as e:
            self.logger.error(f"[MLModel] Error during training classical models - {e}")

    def train_pytorch(self, X_train, y_train, X_val, y_val):
        """
        Train the PyTorch neural network.
        """
        self.train_pytorch_model(X_train, y_train, X_val, y_val)
        self.save_pytorch_model()

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train all models (SGD, RF, GB, Voting, PyTorch) and save them.
        """
        if X_train.size == 0 or y_train.size == 0:
            self.logger.warning("[MLModel] Empty training data. Skipping training.")
            return

        try:
            # Train classical models
            self.train_classical_models(X_train, y_train)

            # Train PyTorch model
            self.train_pytorch(X_train, y_train, X_val, y_val)

            # Save scaler and PCA after training
            self.save_scaler_pca()

            self.logger.info("[MLModel] All models trained and saved successfully.")
        except Exception as e:
            self.logger.error(f"[MLModel] Error during overall training - {e}")

    def predict_pytorch(self, X):
        """
        Use the PyTorch model to predict (binary 0/1).
        """
        if X.shape[0] == 0:
            self.logger.error("[MLModel] Empty input array. Cannot perform prediction.")
            return None

        try:
            self.pytorch_model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                preds = self.pytorch_model(X_tensor)
                predictions = (preds > 0.5).float().cpu().numpy()
                self.logger.info("[MLModel] PyTorch predictions computed successfully.")
                return predictions
        except Exception as e:
            self.logger.error(f"[MLModel] PyTorch prediction failed - {e}")
            return None

    def predict_classical_models(self, X):
        """
        Predict using classical ML models.
        """
        try:
            sgd_pred = self.models["sgd"].predict(X)
            rf_pred = self.models["rf"].predict(X)
            gb_pred = self.models["gb"].predict(X)
            voting_pred = self.voting_model.predict(X)
            self.logger.info("[MLModel] Classical ML models predictions computed successfully.")
            return sgd_pred, rf_pred, gb_pred, voting_pred
        except Exception as e:
            self.logger.error(f"[MLModel] Classical models prediction failed - {e}")
            return None, None, None, None

    def predict(self, X):
        """
        Ensemble prediction from [sgd, rf, gb, voting, pytorch].
        Majority voting is used to determine the final prediction.
        """
        if X.shape[0] == 0:
            self.logger.error("[MLModel] No data provided for prediction.")
            return None

        try:
            # Classical models predictions
            sgd_pred, rf_pred, gb_pred, voting_pred = self.predict_classical_models(X)
            if None in [sgd_pred, rf_pred, gb_pred, voting_pred]:
                self.logger.error("[MLModel] One or more classical models failed to predict.")
                return None

            # PyTorch predictions
            pytorch_pred = self.predict_pytorch(X)
            if pytorch_pred is None:
                self.logger.warning("[MLModel] PyTorch prediction failed. Using classical models only.")
                ensemble_preds = np.vstack((sgd_pred, rf_pred, gb_pred, voting_pred))
            else:
                ensemble_preds = np.vstack((sgd_pred, rf_pred, gb_pred, voting_pred, pytorch_pred.flatten()))

            # Majority voting
            final_pred = np.apply_along_axis(lambda row: np.bincount(row.astype(int)).argmax(), axis=0, arr=ensemble_preds)
            self.logger.info(f"[MLModel] Ensemble final prediction computed successfully.")
            return final_pred
        except Exception as e:
            self.logger.error(f"[MLModel] Ensemble prediction failed - {e}")
            return None

    def auto_feature_engineering(self, price_data):
        """
        Optional: Use TSFresh for advanced feature extraction.
        """
        self.logger.info("[MLModel] Starting TSFresh feature extraction.")
        try:
            df = pd.DataFrame(price_data['candles'])
            df['id'] = 0  # Single time series
            df['time'] = pd.to_datetime(df['time'])
            features = extract_features(df, column_id="id", column_sort="time")
            self.logger.info(f"[MLModel] TSFresh extracted {features.shape[1]} features.")
            return features.fillna(0).values
        except Exception as e:
            self.logger.error(f"[MLModel] TSFresh feature extraction failed - {e}")
            return np.empty((0,))

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the performance of all models on test data.
        """
        self.logger.info("[MLModel] Starting model evaluation.")
        try:
            # Classical models evaluation
            for model_name, model in self.models.items():
                try:
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    self.logger.info(f"[MLModel] {model_name.upper()} -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
                except Exception as e:
                    self.logger.error(f"[MLModel] Evaluation failed for {model_name.upper()} - {e}")

            # Voting classifier evaluation
            try:
                y_pred_voting = self.voting_model.predict(X_test)
                acc_voting = accuracy_score(y_test, y_pred_voting)
                prec_voting = precision_score(y_test, y_pred_voting, zero_division=0)
                rec_voting = recall_score(y_test, y_pred_voting, zero_division=0)
                self.logger.info(f"[MLModel] VOTING CLASSIFIER -> Accuracy: {acc_voting:.4f}, Precision: {prec_voting:.4f}, Recall: {rec_voting:.4f}")
            except Exception as e:
                self.logger.error(f"[MLModel] Evaluation failed for Voting Classifier - {e}")

            # PyTorch model evaluation
            try:
                y_pred_pytorch = self.predict_pytorch(X_test)
                if y_pred_pytorch is not None:
                    y_pred_pytorch = y_pred_pytorch.flatten()
                    acc_pytorch = accuracy_score(y_test, y_pred_pytorch)
                    prec_pytorch = precision_score(y_test, y_pred_pytorch, zero_division=0)
                    rec_pytorch = recall_score(y_test, y_pred_pytorch, zero_division=0)
                    self.logger.info(f"[MLModel] PYTORCH MODEL -> Accuracy: {acc_pytorch:.4f}, Precision: {prec_pytorch:.4f}, Recall: {rec_pytorch:.4f}")
                else:
                    self.logger.warning("[MLModel] PyTorch model did not produce predictions.")
            except Exception as e:
                self.logger.error(f"[MLModel] Evaluation failed for PyTorch model - {e}")
        except Exception as e:
            self.logger.error(f"[MLModel] Model evaluation failed - {e}")

    def get_success_rate(self):
        """
        Calculate the success rate of trades based on internal stats.
        """
        total = self.successful_trades + self.failed_trades
        if total == 0:
            return 0.0
        return round((self.successful_trades / total) * 100, 2)

    def get_trade_stats(self):
        """
        Return trade statistics.
        """
        return {
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "success_rate": self.get_success_rate()
        }
