from pathlib import Path
import sys
import types

import pytest

pd = pytest.importorskip("pandas")

from chain_of_alpha_nasdaq import data_handler

FIXTURE_DIR = Path("tests/fixtures/ohlcv")


def test_fetch_data_local_fixture():
    df = data_handler.fetch_data(
        ["AAPL", "MSFT"],
        start="2020-01-02",
        end="2020-01-06",
        source="local",
        local_dir=FIXTURE_DIR,
    )

    assert not df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.min().date().isoformat() == "2020-01-02"
    assert df.index.max().date().isoformat() == "2020-01-06"

    # MultiIndex columns retain ticker/field separation for large universes
    assert isinstance(df.columns, pd.MultiIndex)
    assert df.columns.names == ["Ticker", "Field"]

    expected_fields = {"Open", "High", "Low", "Close", "Volume"}
    assert set(df.columns.get_level_values("Field")) == expected_fields
    assert {"AAPL", "MSFT"}.issubset(set(df.columns.get_level_values("Ticker")))

    # Ensure the schema is numeric and without duplicate indices
    assert df.index.is_monotonic_increasing
    assert df.apply(pd.to_numeric, errors="coerce").notna().all().all()


def test_fetch_data_falls_back_to_yfinance(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "dummy")

    fixture_map = {
        ticker: pd.read_csv(FIXTURE_DIR / f"{ticker}.csv")
        for ticker in ("AAPL", "MSFT")
    }

    def fake_polygon(*_, **__):
        raise RuntimeError("polygon down")

    def fake_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
        df = fixture_map[ticker].copy()
        date_column = "Date" if "Date" in df.columns else "date"
        mask = (df[date_column].astype(str) >= start) & (df[date_column].astype(str) <= end)
        df = df.loc[mask].copy()
        if date_column != "Date":
            df.rename(columns={date_column: "Date"}, inplace=True)
        return df

    monkeypatch.setattr(data_handler, "fetch_polygon", fake_polygon)
    monkeypatch.setattr(data_handler, "_fetch_yfinance", fake_yfinance)

    df = data_handler.fetch_data(
        ["AAPL", "MSFT"],
        start="2020-01-02",
        end="2020-01-03",
        local_dir=FIXTURE_DIR,
    )

    assert not df.empty
    assert ("AAPL", "Open") in df.columns
    assert df.loc["2020-01-02", ("MSFT", "Close")] == pytest.approx(160.619995)


def test_fetch_data_resolves_nasdaq_universe(monkeypatch):
    call_state = {"called": False}

    module = types.ModuleType("ticker_fetcher")

    def fake_fetch():
        call_state["called"] = True
        return pd.DataFrame({"ticker": ["AAPL", "MSFT", "GOOG"]})

    module.fetch_nasdaq_tickers = fake_fetch

    monkeypatch.setitem(
        sys.modules,
        "chain_of_alpha_nasdaq.utils.ticker_fetcher",
        module,
    )

    df = data_handler.fetch_data(
        "nasdaq",
        start="2020-01-02",
        end="2020-01-03",
        source="local",
        local_dir=FIXTURE_DIR,
    )

    assert call_state["called"] is True
    assert isinstance(df.columns, pd.MultiIndex)
    assert "AAPL" in df.columns.get_level_values("Ticker")


def test_fetch_data_nasdaq_fallback(monkeypatch):
    failing_module = types.ModuleType("ticker_fetcher")

    def failing_fetch():
        raise RuntimeError("polygon unavailable")

    failing_module.fetch_nasdaq_tickers = failing_fetch

    monkeypatch.setitem(
        sys.modules,
        "chain_of_alpha_nasdaq.utils.ticker_fetcher",
        failing_module,
    )

    call_state = {"requested": False}

    class DummyResponse:
        status_code = 200

        def __init__(self, text):
            self.text = text

    sample_payload = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc.|Q|N|N|100|N|N
MSFT|Microsoft Corp.|Q|N|N|100|N|N
"""

    def fake_get(url, timeout):
        call_state["requested"] = True
        return DummyResponse(sample_payload)

    monkeypatch.setattr(data_handler.requests, "get", fake_get)

    df = data_handler.fetch_data(
        "nasdaq",
        start="2020-01-02",
        end="2020-01-03",
        source="local",
        local_dir=FIXTURE_DIR,
    )

    assert call_state["requested"] is True
    assert isinstance(df.columns, pd.MultiIndex)
    assert {"AAPL", "MSFT"}.issubset(df.columns.get_level_values("Ticker"))
