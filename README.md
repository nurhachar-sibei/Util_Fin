# 金融分析工具集（Util_Fin）

## 项目简介

Util_Fin 是一套围绕投资研究、组合构建与数据管理打包的实用工具。从最初的波动率估计脚本逐步发展成涵盖数据获取、风险建模、策略评估、调仓管理以及数据库运维的综合工具箱。所有模块彼此解耦、可单独调用，也可以按“获取数据 ➝ 处理建模 ➝ 回测评估 ➝ 入库归档”的流程串联使用，帮你快速搭建贴合自身习惯的量化研究环境。

## 模块总览

| 模块 | 功能概述 | 典型场景 |
| --- | --- | --- |
| `Wind_util.py` | WindPy 数据抓取、行情更新、绘图与收益率生成 | 批量拉取/维护资产价格库 |
| `Volatility_util.py` | 9 种协方差与半协方差模型 | 风险建模、组合优化、风险平价 |
| `PCAanalysis.py` | 标准化、PCA 拟合、指标导出与可视化 | 因子降维、相关性诊断 |
| `eval_module.py` | 绩效、风险、年度/月度分析与报表 | 策略评估、风控复盘 |
| `Eval_util.py` | 旧版评估函数（保留兼容） | 与旧脚本兼容 |
| `Position_util.py` | 固定日期/间隔调仓计划生成 | 回测调仓日管理 |
| `Dynamic_weight.py` | 权重随行情漂移的动态回溯 | 权重风控、业绩归因 |
| `EWMA_weight_show.py` | EWMA 权重衰减分析与可视化 | 设定 λ 的感性校验 |
| `RP_solo_ver2_3.py` | 风险平价模型、回测、可视化与评估 | 多资产风控配置 |
| `easy_manager.py` | PostgreSQL 数据入库、扩列、查询，含 LongManager | 研究数据仓库、面板数据管理 |
| `logger_util.py` | 多 logger 快速配置与复用 | 为脚本统一落盘日志 |

## 主要功能详解

### 数据获取：Wind_util.py
- 支持一次性拉取多个证券的前复权收盘价、任意日度字段，并转换为 `DataFrame`。
- `update_hfq_price`/`add_new_asset` 可直接维护本地 Excel 价格库。
- 内置线图、直方图和相关性热力图，便于快速检查数据质量。

```python
from Wind_util import get_hfq_price, get_return_df

codes = ['000300.SH', '000905.SH']
price_df = get_hfq_price(codes, '2018-01-01', '2024-12-31')
ret_df = get_return_df(price_df)
```

### 波动率与风险模型：Volatility_util.py
- `Cov_Matrix` 封装 9 种风险模型（样本、半衰、Ledoit-Wolf、EWMA/GARCH、半方差等），并支持设置频率与 λ。
- 统一接口 `calculate_cal_cov_matrix`，便于在任意策略里切换风险模型。

### 降维分析：PCAanalysis.py
- 自动标准化、缺失校验、主成分拟合、方差解释指标导出。
- 提供方差解释图、载荷热力图、双标图等可视化方法，并支持 `export_results` 导出 Excel。

### 策略绩效与风险评估：eval_module.py（推荐）
- `PerformanceEvaluator`：一次性输出累计收益、年化收益/波动、最大回撤、Sharpe/Calmar/Sortino 等指标。
- `RiskAnalyzer`：提供 VaR/CVaR、滚动回撤、分位压力测试。
- `PeriodAnalyzer`：按年、月拆分风险收益，同时支持基于历史窗口的 VaR 预测与可投规模估算。
- `ReportGenerator`（见文件后半部分）：生成整合图表和 Excel/markdown 报告。
- 老版本 `Eval_util.py` 仍保留，可在遗留脚本中引用。

```python
from eval_module import PerformanceEvaluator, RiskAnalyzer, PeriodAnalyzer

pv = strategy_nav['nv']
dashboard = PerformanceEvaluator.evaluate_portfolio(pv, name='MyStrategy')
rolling_dd = RiskAnalyzer.calculate_rolling_drawdown(pv.pct_change().fillna(0))
annual = PeriodAnalyzer.annual_analysis(pv.pct_change().fillna(0))
```

### 组合管理与回测

- **Position_util.py**：按固定交易日间隔、指定年月日、或 Pandas 频率字符串（如 `'M'/'W-FRI'`）生成调仓日，返回整段收益、调仓收益与日期列表。
- **Dynamic_weight.py**：根据单期收益和初始权重模拟“价格漂移”后的真实权重，可配合调仓日序列批量计算。
- **EWMA_weight_show.py**：分析不同 λ、窗口下的 EWMA 权重，带线性/对数坐标对比、热力图以及权重集中度分析。
- **RP_solo_ver2_3.py**：`RPmodel` 融合 Wind 数据、协方差模型、风险预算算法（普通/基于 PCA/基于 LASSO）、杠杆与现金成本设置，并包含：
  - 调仓规则生成、动态权重调整
  - 回测与基准对比、可视化（权重演化、净值曲线）
  - 绩效评估（调用 `Eval_util`/`eval_module`）

```python
from Dynamic_weight import calculate_dynamic_weight
daily_weights = calculate_dynamic_weight(ret_df, init_weight=init_w, init_date=ret_df.index[0])

from RP_solo_ver2_3 import RPmodel
model = RPmodel(ret_df, start_date='2021-01-01', end_date='2024-12-31',
                cov_matrix_method='EWMA_SEMI', change_time_delta=20)
model.position_get(M=1, D=1)
model.Backtest()
model.get_eval()
```

### 数据库与基础设施

- **easy_manager.py**
  - `EasyManager`：连接 PostgreSQL、根据 `DataFrame` 自动建表、推断字段类型、批量写入、扩列、去重插入、表结构查询等。
  - `LongManager`：继承自 `EasyManager`，专门处理（时间, 实体）面板数据，内置复合索引、`skip/update/append` 三种增量写入模式。
- **logger_util.py**：`setup_logger` 统一创建 console+file 的多 logger，配合 `easy_manager` 的 `function_timer` 装饰器记录运行耗时。

```python
from easy_manager import EasyManager, LongManager
import pandas as pd

with EasyManager(database='research') as em:
    em.create_table('factor_snapshot', df, overwrite=True)
    em.insert_data('factor_snapshot', new_df, mode='upsert')

with LongManager(time_col='trade_date', entity_col='ticker') as lm:
    lm.create_table('panel_factor', panel_df)
```

## 安装依赖

核心依赖：
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn arch pypfopt tqdm tabulate
```

可选依赖：
- `WindPy`：需安装万得终端并配置 Python 接口。
- `psycopg2`：`pip install psycopg2-binary`（PostgreSQL 数据库连接）。
- `pywin32` 等 Windows 组件：用于 WindPy/Excel 交互。

## 快速上手

### 协方差矩阵
```python
from Volatility_util import Cov_Matrix

cov = Cov_Matrix(ret_df, method='EWMA').calculate_cal_cov_matrix(frequency=252, lambda_=0.94)
```

### PCA 分析
```python
from PCAanalysis import PCAAnalyzer

pca = PCAAnalyzer(standardize=True).fit(ret_df)
pca.plot_components_heatmap()
pca.export_results('pca_results.xlsx')
```

### 策略评估
```python
from eval_module import PerformanceEvaluator, ReportGenerator

report = PerformanceEvaluator.evaluate_multi_portfolios({'Alpha': nav_alpha, 'Beta': nav_beta})
ReportGenerator(report, output_dir='./report').export_excel()
```

### 风险平价回测
```python
rp = RPmodel(ret_df, change_time_delta=15, cov_matrix_method='SPRING', risk_budget_objective='naive_risk_parity')
rp.position_get(M=1, D=5)
rp.Backtest()
rp.plot_pv({'RP': rp.result_rp, 'Benchmark': rp.result_benchmark})
```

### 数据入库
```python
with EasyManager(database='macro_data_base') as em:
    em.create_table('macro_raw', macro_df, overwrite=True)
    em.add_columns('macro_raw', extra_df)
    snapshot = em.load_table('macro_raw', limit=1000)
```

### EWMA 权重可视化
```python
from EWMA_weight_show import EWMAWeightVisualizer
viz = EWMAWeightVisualizer()
viz.plot_multiple_lambda_weights(n_days=120, lambda_values=[0.86, 0.92, 0.97])
```

## 项目结构

```
Util_Fin/
├── Dynamic_weight.py       # 权重随行情漂移
├── easy_manager.py         # PostgreSQL 管理器（EasyManager / LongManager）
├── eval_module.py          # 新版策略评估
├── Eval_util.py            # 兼容旧版评估
├── EWMA_weight_show.py     # EWMA 权重展示
├── logger_util.py          # 日志配置
├── PCAanalysis.py          # PCA 分析工具
├── Position_util.py        # 调仓日管理
├── RP_solo_ver2_3.py       # 风险平价模型
├── Volatility_util.py      # 协方差/风险模型
├── Wind_util.py            # Wind 接口与可视化
├── README.md               # 中文文档
└── README_EN.md            # 英文文档
```

## 版本历史

- **v5.0**：新增 `easy_manager`、`eval_module`、`Dynamic_weight`、`EWMA_weight_show`、`logger_util` 等模块，完善风险平价模型。
- **v4.0**：加入策略评估（旧版）与仓位管理功能。
- **v3.0**：引入 PCA 分析，形成完整分析链路。
- **v2.0**：扩展多种协方差模型。
- **v1.0**：基础波动率计算工具。

## 应用场景

- **风险管理**：多模型 VaR、滚动回撤、投资规模测算。
- **资产配置/风险平价**：协方差建模 + RPmodel + 动态权重分析。
- **量化研究**：PCA 降维、策略回测评估、EWMA 权重可视化。
- **数据工程**：Wind 数据抓取、PostgreSQL 入库、日志留存。

通过组合这些模块，可以快速搭建“数据入库 ➝ 分析建模 ➝ 回测评估”的闭环流程，并根据需要替换或扩展任意步骤。
