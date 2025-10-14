# chain_of_alpha_nasdaq/broker/paper_broker.py
import pandas as pd
from datetime import datetime
from chain_of_alpha_nasdaq.sqlite_store import SQLiteStore

class PaperBroker:
    """
    A minimal simulated broker for paper trading.
    Logs every action into SQLite (paper_trades table).
    """

    def __init__(self, store: SQLiteStore, run_id: str, starting_cash: float = 100_000.0):
        self.store = store
        self.run_id = run_id
        self.cash = starting_cash
        self.positions = {}  # {ticker: quantity}
        self.last_prices = {}  # for equity computation
        print(f"🏦 PaperBroker initialized with ${self.cash:,.2f} cash")

    # ------------------------------------------------------------------
    # Core Execution
    # ------------------------------------------------------------------
    def execute_signal(self, ticker: str, signal: float, price: float):
        """
        Converts a numeric signal (-1 to +1) into a trade.
        Positive signal → BUY, negative signal → SELL.
        """

        # Store last seen price for equity tracking
        self.last_prices[ticker] = price

        # Decision thresholds (configurable later)
        buy_threshold = 0.25
        sell_threshold = -0.25

        if signal > buy_threshold:
            action = "BUY"
            qty = (self.cash * 0.02) / price  # use 2% of cash per buy
            cost = qty * price
            if self.cash >= cost:
                self.cash -= cost
                self.positions[ticker] = self.positions.get(ticker, 0) + qty
            else:
                print(f"⚠️ Not enough cash to buy {ticker}")
                return
        elif signal < sell_threshold:
            action = "SELL"
            qty = self.positions.get(ticker, 0)
            if qty <= 0:
                print(f"⚠️ No position to sell for {ticker}")
                return
            self.cash += qty * price
            self.positions[ticker] = 0
        else:
            action = "HOLD"
            qty = 0

        if action != "HOLD":
            self.store.insert_paper_trade(ticker, action, qty, price, self.run_id)
            print(f"🧾 {action} {qty:.2f} {ticker} @ {price:.2f}")

    # ------------------------------------------------------------------
    # Portfolio Calculations
    # ------------------------------------------------------------------
    def get_equity(self):
        """
        Returns current total equity = cash + market value of positions.
        """
        value = sum(q * self.last_prices.get(t, 0) for t, q in self.positions.items())
        return self.cash + value

    def portfolio_snapshot(self):
        """
        Returns a summary DataFrame of all current holdings.
        """
        df = pd.DataFrame([
            {"ticker": t, "quantity": q, "price": self.last_prices.get(t, 0), "value": q * self.last_prices.get(t, 0)}
            for t, q in self.positions.items()
        ])
        df["cash"] = self.cash
        df["equity"] = self.get_equity()
        return df

    def summary(self):
        print("\n📊 PAPER BROKER SUMMARY")
        print(f"💰 Cash: ${self.cash:,.2f}")
        print(f"💼 Positions: {self.positions}")
        print(f"📈 Total equity: ${self.get_equity():,.2f}")