OANDA_API_KEY = "f937aebe528d10700196dc09b9259322-c79e8fafd058bdf438e740223701a32a"
OANDA_ACCOUNT_ID = "101-001-27648896-001"
BASE_URL = "https://api-fxpractice.oanda.com/v3"
TRADE_INSTRUMENT = "EUR_USD"  # Example instrument

# PyTorch-specific parameters
PYTORCH_HIDDEN_SIZE = 64  # Hidden layer size for the PyTorch model
PYTORCH_LEARNING_RATE = 0.001  # Learning rate for the optimizer
PYTORCH_EPOCHS = 10  # Number of training epochs

# Model Configuration
MODEL_CONFIG = {
    'input_size': 10,         # Number of input features (adjust based on your feature engineering)
    'hidden_size': PYTORCH_HIDDEN_SIZE,
    'output_size': 1,         # For binary classification (e.g., buy/sell)
    'learning_rate': PYTORCH_LEARNING_RATE,
    'epochs': PYTORCH_EPOCHS,
    # Add other model-related configurations if necessary
}