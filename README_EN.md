# Financial Analysis Toolkit (Util_Fin)

## Overview

Util_Fin is a collection of reusable utilities that cover the full workflow of quantitative research: fetching market data, building risk models, evaluating strategies, managing rebalancing rules, and persisting results in a database. Each module can run independently, yet they work best when chained together to form a “data ➝ modeling ➝ backtest ➝ evaluation ➝ storage” loop tailored to your own process.

## Module Catalog

| Module | Highlights | Typical Use |
| --- | --- | --- |
| `Wind_util.py` | Wrapper around WindPy for OHLC/factor download, local cache maintenance, quick plots | Build/refresh local price libraries |
| `Volatility_util.py` | 9 covariance / semi-covariance models with unified interface | Risk modeling, portfolio optimization, risk parity |
| `PCAanalysis.py` | PCA with standardization, validation, metrics, and visualizations | Factor reduction, correlation diagnostics |
| `eval_module.py` | Performance dashboard, risk metrics, annual/monthly drill-down, report export | Strategy evaluation, risk review |
| `Eval_util.py` | Legacy evaluation helpers (kept for compatibility) | Legacy scripts |
| `Position_util.py` | Generate calendar-based or fixed-interval rebalancing dates | Backtest scheduling |
| `Dynamic_weight.py` | Track how weights drift with price moves | Weight risk control, attribution |
| `EWMA_weight_show.py` | Analyze/visualize EWMA weight decay under different λ | Tune decay parameters |
| `RP_solo_ver2_3.py` | Full risk-parity modeling, backtest, visualization, evaluation | Multi-asset allocation |
| `easy_manager.py` | PostgreSQL helper (EasyManager) plus panel-data oriented LongManager | Research data warehouse |
| `logger_util.py` | Shared logger factory (console + file) | Consistent logging across scripts |

## Feature Highlights

### Data acquisition – `Wind_util.py`
- Batch download multiple tickers and factors from Wind, returning clean `DataFrame`s.
- `update_hfq_price` and `add_new_asset` maintain Excel-based price stores.
- Built-in line/hist/heatmap plots for quick QA.

```python
from Wind_util import get_hfq_price, get_return_df

codes = ['000300.SH', '000905.SH']
prices = get_hfq_price(codes, '2018-01-01', '2024-12-31')
returns = get_return_df(prices)
```

### Volatility & risk models – `Volatility_util.py`
- `Cov_Matrix` exposes sample, half-life, diagonal, Ledoit-Wolf shrinkage, threshold, EWMA, GARCH, (EWMA) semi-covariance and more.
- All models share the method `calculate_cal_cov_matrix`, making it easy to swap risk models inside any optimizer or backtest.

### Dimensionality reduction – `PCAanalysis.py`
- Handles data validation, standardization, PCA fitting, explained-variance metrics, component heatmaps/biplots, and Excel export.

### Strategy evaluation – `eval_module.py`
- `PerformanceEvaluator`: cumulative return, annual return/vol, max drawdown, Sharpe, Calmar, Sortino, terminal NAV, multi-portfolio comparison.
- `RiskAnalyzer`: historical/parametric VaR, CVaR, rolling drawdown, risk sampling helpers.
- `PeriodAnalyzer`: yearly/monthly breakdown with historical-window VaR forecasts and investable-capital estimation.
- `ReportGenerator`: consolidate tables/figures into Excel or Markdown reports.
- The legacy `Eval_util.py` remains for older notebooks/scripts.

```python
from eval_module import PerformanceEvaluator, RiskAnalyzer, PeriodAnalyzer

nav = strategy_df['nav']
summary = PerformanceEvaluator.evaluate_portfolio(nav, name='Alpha')
rolling = RiskAnalyzer.calculate_rolling_drawdown(nav.pct_change().fillna(0))
annual = PeriodAnalyzer.annual_analysis(nav.pct_change().fillna(0))
```

### Portfolio management & backtesting

- **`Position_util.py`**: generate rebalancing dates by stride (e.g., every 20 trading days), by explicit month/day, or by pandas frequency strings such as `'M'` and `'W-FRI'`.
- **`Dynamic_weight.py`**: simulate how weights drift between rebalancing dates and stitch multiple windows together.
- **`EWMA_weight_show.py`**: inspect decay curves under various λ, plot weight heatmaps, measure concentration/effective lookback.
- **`RP_solo_ver2_3.py`**: `RPmodel` glues Wind data, covariance engines, risk-budget solvers (naive, PCA-based, LASSO-based), leverage/cash-cost settings, analytics, visualization, and evaluation.

```python
from Dynamic_weight import calculate_dynamic_weight
daily_weight = calculate_dynamic_weight(hold_df=ret_df, init_weight=init_w, init_date=ret_df.index[0])

from RP_solo_ver2_3 import RPmodel
rp = RPmodel(ret_df, start_date='2021-01-01', end_date='2024-12-31',
             cov_matrix_method='EWMA_SEMI', change_time_delta=20)
rp.position_get(M=1, D=1)
rp.Backtest()
rp.get_eval()
```

### Data infrastructure – `easy_manager.py` & `logger_util.py`

- **EasyManager**: connect to PostgreSQL, infer column types from `DataFrame`s, create tables, batch insert, add columns, deduplicate inserts (`mode='skip'/'update'/'append'`), load tables, inspect schemas. Comes with a `function_timer` decorator that logs runtime through `logger_util`.
- **LongManager**: inherits EasyManager, enforces (time, entity) composite keys for panel data, auto-creates multi-column indexes, and ensures time column comes first.
- **logger_util**: `setup_logger` builds reusable console/file loggers so every tool writes consistent audit trails.

```python
from easy_manager import EasyManager, LongManager

with EasyManager(database='research') as em:
    em.create_table('factor_snapshot', df, overwrite=True)
    em.insert_data('factor_snapshot', incr_df, mode='update')

with LongManager(time_col='trade_date', entity_col='ticker') as lm:
    lm.create_table('factor_panel', panel_df)
```

## Dependencies

Install the common Python stack:

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn arch pypfopt tqdm tabulate
```

Optional/extra:
- `WindPy` (from the Wind terminal) for `Wind_util.py`.
- `psycopg2-binary` for PostgreSQL access.
- `pywin32` and other Windows components when interacting with Excel/Wind.

## Quick Start Recipes

### Covariance matrix
```python
from Volatility_util import Cov_Matrix

cov = Cov_Matrix(ret_df, method='EWMA').calculate_cal_cov_matrix(frequency=252, lambda_=0.94)
```

### PCA
```python
from PCAanalysis import PCAAnalyzer

pca = PCAAnalyzer(standardize=True).fit(ret_df)
pca.plot_explained_variance()
pca.export_results('pca_results.xlsx')
```

### Performance report
```python
from eval_module import PerformanceEvaluator, ReportGenerator

dashboard = PerformanceEvaluator.evaluate_multi_portfolios({
    'Alpha': nav_alpha, 'Beta': nav_beta
})
ReportGenerator(dashboard, output_dir='./report').export_excel()
```

### Risk-parity backtest
```python
from RP_solo_ver2_3 import RPmodel

rp = RPmodel(ret_df, change_time_delta=15, cov_matrix_method='SPRING')
rp.position_get(M=1, D=5)
rp.Backtest()
rp.plot_pv({'RP': rp.result_rp, 'BM': rp.result_benchmark})
```

### Database ingestion
```python
with EasyManager(database='macro_data_base') as em:
    em.create_table('macro_raw', macro_df, overwrite=True)
    em.add_columns('macro_raw', extra_df)
    snapshot = em.load_table('macro_raw', limit=1000)
```

### EWMA weight visualization
```python
from EWMA_weight_show import EWMAWeightVisualizer

viz = EWMAWeightVisualizer()
viz.plot_multiple_lambda_weights(n_days=120, lambda_values=[0.86, 0.92, 0.97])
```

## Project Structure

```
Util_Fin/
├── Dynamic_weight.py       # Weight drift calculator
├── easy_manager.py         # PostgreSQL helpers (EasyManager / LongManager)
├── eval_module.py          # Modern evaluation toolkit
├── Eval_util.py            # Legacy evaluation helpers
├── EWMA_weight_show.py     # EWMA weight visualization
├── logger_util.py          # Logger factory
├── PCAanalysis.py          # PCA analyzer
├── Position_util.py        # Rebalancing calendar manager
├── RP_solo_ver2_3.py       # Risk-parity engine
├── Volatility_util.py      # Covariance/risk models
├── Wind_util.py            # Wind data interface
├── README.md               # Chinese documentation
└── README_EN.md            # English documentation
```

## Version History

- **v5.0**: Added database helpers, the new evaluation module, weight-visualization tools, logger utilities, and refreshed the risk-parity engine.
- **v4.0**: Introduced strategy evaluation (legacy) and position management.
- **v3.0**: Added PCA analysis to complete the research pipeline.
- **v2.0**: Extended the volatility toolkit with multiple covariance estimators.
- **v1.0**: Initial volatility calculator.

## Use Cases

- **Risk management**: multi-model VaR, rolling drawdown dashboards, investable-capital estimation.
- **Asset allocation / risk parity**: covariance modeling + RPmodel + dynamic-weight inspection.
- **Quant research**: PCA for factor compression, backtest evaluation, EWMA decay diagnostics.
- **Data engineering**: fetch from Wind, store/update in PostgreSQL, keep unified logs.

Mix any subset of the modules to create your own workflow, and plug in additional analytics when needed.
