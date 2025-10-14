import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# API Keys
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Trading / Backtest settings
DATA_SOURCE = "yfinance"  # fallback if no API keys provided
TICKER_UNIVERSE = "NASDAQ"
START_DATE = "2015-01-01"
END_DATE = None

TOP_K = 50
TRANSACTION_COST = {"buy": 0.0003, "sell": 0.0010}
REBALANCE_FREQ = "1d"

MAX_FACTORS = 100
OPTIMIZATION_STEPS = 5