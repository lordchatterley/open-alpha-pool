import pytest

pd = pytest.importorskip("pandas")

from chain_of_alpha_nasdaq.factor_optimization import FactorOptimizer


def _make_price_panel():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    tickers = ["AAA", "BBB"]
    fields = ["Open", "High", "Low", "Close", "Volume"]
    columns = pd.MultiIndex.from_product([tickers, fields], names=["Ticker", "Field"])
    frame = pd.DataFrame(index=dates, columns=columns, dtype=float)

    base_prices = {
        "AAA": [10, 10.5, 11.0, 11.5, 12.0, 12.5],
        "BBB": [20, 19.5, 19.0, 18.5, 18.0, 17.5],
    }

    for ticker in tickers:
        close = pd.Series(base_prices[ticker], index=dates)
        frame[(ticker, "Close")] = close
        frame[(ticker, "Open")] = close.shift(1).fillna(close.iloc[0])
        frame[(ticker, "High")] = close * 1.01
        frame[(ticker, "Low")] = close * 0.99
        frame[(ticker, "Volume")] = 1_000_000

    return frame


def test_optimize_factor_returns_cross_sectional_metrics():
    data = _make_price_panel()
    optimizer = FactorOptimizer(horizon=1)

    result = optimizer.optimize_factor(
        factor_name="alpha_test",
        formula="Close.pct_change().shift(-1)",
        data=data,
    )

    assert result is not None
    assert set(result.signals.columns) == {"AAA", "BBB"}
    assert result.forward_returns.index.equals(result.signals.index)
    assert result.metrics["Coverage"] > 0.0
    assert result.metrics["Observations"] > 0
    # Signals align with forward returns, so IC and RankIC should be near 1
    assert result.metrics["IC"] > 0.9
    assert result.metrics["RankIC"] > 0.9


def test_optimize_factor_penalises_low_diversity():
    data = _make_price_panel()
    optimizer = FactorOptimizer(horizon=1)

    baseline = optimizer.optimize_factor(
        factor_name="alpha_base",
        formula="Close.pct_change().shift(-1)",
        data=data,
    )
    assert baseline is not None

    follow_up = optimizer.optimize_factor(
        factor_name="alpha_clone",
        formula="Close.pct_change().shift(-1)",
        data=data,
        existing_signals=[baseline.signals],
    )

    assert follow_up is not None
    assert follow_up.metrics["Diversity"] <= 0.1
