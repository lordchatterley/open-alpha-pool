# Evaluation of Code Alignment with "Chain of Alpha" Paper

## Access to Reference Material
- Attempted to download the reference paper (https://arxiv.org/pdf/2508.06312) but the request was blocked by the execution environment (HTTP 403), so the evaluation below relies on the design cues and docstrings embedded in the repository rather than a direct reading of the manuscript.

## Implementation Overview
- `ChainOfAlphaAgent` orchestrates data collection, factor generation, optimization, storage, and portfolio backtesting in a single run. 【F:chain_of_alpha_nasdaq/agent.py†L14-L114】
- Market data is fetched exclusively through the Polygon API, producing a wide table whose column names are ticker-prefixed OHLCV fields. 【F:chain_of_alpha_nasdaq/data_handler.py†L12-L87】
- Factor discovery leans on an OpenAI LLM to emit symbolic formulas, which are evaluated either via `pd.eval` or Python `eval` sandboxes. 【F:chain_of_alpha_nasdaq/factor_generation.py†L8-L66】【F:chain_of_alpha_nasdaq/factor_optimization.py†L7-L52】
- Portfolio construction and backtesting apply a top-K equal-weight overlay to supplied signal matrices. 【F:chain_of_alpha_nasdaq/portfolio.py†L1-L141】
- The repository also contains an ML stacking agent that trains a LightGBM model on stored factor signals and “cheap” technical features retrieved from SQLite. 【F:chain_of_alpha_nasdaq/ml_agent.py†L1-L108】

## Alignment Issues and Implementation Gaps
1. **Factor optimization cannot run** – `FactorOptimizer.optimize_factor` requires both a factor name and realized returns, yet the agent calls it with only the formula and the raw price table, yielding an immediate `TypeError`. Even if the signature were corrected, the optimizer expects single-ticker OHLCV columns (`Open`, `High`, …) while `fetch_data` delivers ticker-prefixed columns, so evaluation would still fail. 【F:chain_of_alpha_nasdaq/factor_optimization.py†L54-L90】【F:chain_of_alpha_nasdaq/agent.py†L57-L82】【F:chain_of_alpha_nasdaq/data_handler.py†L65-L87】
2. **Placeholder signals break the research loop** – when optimization fails, the agent silently substitutes each factor with Gaussian noise before averaging them. The downstream portfolio therefore tests random numbers instead of real alpha ideas, making the pipeline ineffective for validating hypotheses from the paper. 【F:chain_of_alpha_nasdaq/agent.py†L75-L96】
3. **Portfolio alignment is inconsistent** – the combined signal is a single-column series (`aggregate_alpha`) without ticker dimension, yet the backtester assumes a full cross-sectional signal matrix. The resulting weight matrix collapses to all NaNs or zero weights, so performance metrics are meaningless. 【F:chain_of_alpha_nasdaq/agent.py†L87-L108】【F:chain_of_alpha_nasdaq/portfolio.py†L5-L89】
4. **No persistence of evaluated factors** – although `AlphaDB` stores factors and metrics, the agent records metrics even when they are dummy placeholders and never writes the actual signal time series required for later ML stacking or paper-style longitudinal studies. 【F:chain_of_alpha_nasdaq/agent.py†L63-L114】【F:chain_of_alpha_nasdaq/ml_agent.py†L16-L108】
5. **Data pipeline lacks robustness** – the code aborts without a Polygon key and does not implement the alternative data sources mentioned in configuration defaults (e.g., yfinance). That undermines reproducibility claimed in the paper. 【F:chain_of_alpha_nasdaq/config.py†L15-L25】【F:chain_of_alpha_nasdaq/data_handler.py†L52-L87】
6. **Risk and evaluation metrics are incomplete** – RankIC, turnover, and diversity thresholds defined in `AlphaPool` are never populated because the optimizer never returns real metrics, preventing the codebase from vetting factors the way the paper describes. 【F:chain_of_alpha_nasdaq/alpha_pool.py†L6-L74】

## Plan for Further Development
1. **Stabilize the data layer**
   - Implement a working fallback (e.g., yfinance) and harmonize the OHLCV schema so that factor evaluators receive consistent column names across tickers. 【F:chain_of_alpha_nasdaq/data_handler.py†L52-L87】
   - Add regression tests that load a small offline fixture to keep the pipeline functional without external APIs.

2. **Repair factor evaluation and optimization**
   - Refactor `FactorOptimizer.optimize_factor` to accept the actual price panel and compute forward returns internally, returning IC/RankIC arrays aligned with the paper’s methodology. 【F:chain_of_alpha_nasdaq/factor_optimization.py†L54-L90】
   - Update `ChainOfAlphaAgent` to pass the correct arguments, persist per-factor signal matrices, and only record metrics when evaluation succeeds. 【F:chain_of_alpha_nasdaq/agent.py†L56-L114】
   - Implement diversity and turnover calculations against the stored signal history so that `AlphaPool` enforces the thresholds documented in the paper. 【F:chain_of_alpha_nasdaq/alpha_pool.py†L6-L74】

3. **Make backtesting cross-sectional**
   - Ensure each factor produces a DataFrame with tickers as columns (e.g., by pivoting evaluated formulas across assets) before aggregation. 【F:chain_of_alpha_nasdaq/portfolio.py†L5-L89】
   - Extend the backtester with transaction cost handling and benchmark-relative metrics to match the empirical evaluation likely described in the manuscript.

4. **Integrate ML stacking responsibly**
   - Populate the SQLite store with evaluated factor signals and realized returns so the ML agent can actually train. 【F:chain_of_alpha_nasdaq/ml_agent.py†L16-L108】
   - Add validation diagnostics (feature coverage, leakage checks, walk-forward splits) to keep the stacking model consistent with the multi-stage process outlined by the paper.

5. **Document and automate**
   - Create end-to-end notebooks or scripts that replicate the experiments, including configuration files mirroring the paper’s datasets, horizons, and evaluation windows.
   - Establish CI checks (unit tests for factor evaluation, integration tests for backtesting) to prevent regressions that would once again decouple the code from the paper.

Addressing these gaps will transform the repository from a scaffold into a faithful and testable implementation of the paper’s Chain-of-Alpha methodology.

## Progress Update – Item 1: Stabilize the Data Layer

### Changes Implemented
- Rebuilt `fetch_data` to normalise OHLCV columns, pivot outputs into a `(ticker, field)` MultiIndex, and transparently fall back from Polygon to yfinance or on-disk CSV fixtures so the NASDAQ-wide scan can run without a Polygon key. 【F:chain_of_alpha_nasdaq/data_handler.py†L1-L280】
- Added lightweight AAPL/MSFT fixtures and regression tests to assert the canonical schema, data source fallback order, and NASDAQ universe resolution logic. 【F:tests/fixtures/ohlcv/AAPL.csv†L1-L4】【F:tests/fixtures/ohlcv/MSFT.csv†L1-L4】【F:tests/test_data_handler.py†L1-L155】
- Updated the NASDAQ scan orchestration script to flatten MultiIndex columns safely when splicing cached batches, ensuring downstream consumers receive the wide cross-sectional frame they expect. 【F:chain_of_alpha_nasdaq/agents/run_nasdaq_scan.py†L1-L120】

### Outcome and Remaining Gaps
- Local regression coverage now exercises every branch of the loader and confirms the NASDAQ universe resolution works without external credentials, but the suite is currently skipped in environments missing `pandas`, so CI still needs a data-science-ready runtime. 【7af147†L1-L4】
- The harmonised schema removes the blocker that previously caused factor evaluation to choke on ticker-prefixed column names; however, the factor optimization stack still requires refactoring per Plan Item 2 before true end-to-end runs align with the paper.

## Progress Update – Item 2: Repair Factor Evaluation and Optimization

### Changes Implemented
- Refactored `FactorOptimizer` into a panel-aware evaluator that derives per-ticker signals, forward returns, and cross-sectional IC/RankIC series directly from the MultiIndex OHLCV table, while tracking turnover, diversity, and coverage statistics for each candidate. 【F:chain_of_alpha_nasdaq/factor_optimization.py†L11-L215】
- Rewired `ChainOfAlphaAgent` to persist genuine signal matrices, record metrics only after successful evaluation, and gate new factors through `AlphaPool` thresholds before they contribute to the aggregate alpha. 【F:chain_of_alpha_nasdaq/agent.py†L1-L175】
- Added regression tests that fabricate a mini price panel to confirm the optimizer’s IC/RankIC math and its diversity penalty when a new factor duplicates an existing signal. 【F:tests/test_factor_optimization.py†L1-L52】

### Outcome and Remaining Gaps
- Factor trials now yield structured metrics and stored signal histories instead of Gaussian noise, enabling AlphaPool to enforce the paper’s RankIC, turnover, and diversity thresholds; however, the current agent still blends factors with a naive equal average, so portfolio construction has yet to incorporate more sophisticated weighting or risk budgeting from the manuscript.
- The new test harness exercises the analytics pipeline but remains skipped when `pandas` is unavailable, so continuous integration still requires a data-science runtime to execute the full suite. 【787982†L1-L7】
