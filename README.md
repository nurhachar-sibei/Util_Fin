# 金融分析工具集 (Util_Fin)

## 项目简介

本项目是一个综合性的金融分析工具集，从最初的波动率计算工具进化而来，现已扩展为包含多种金融分析功能的完整工具包。项目包含协方差矩阵计算、主成分分析(PCA)、策略评价分析和仓位管理四大核心模块，适用于金融风险管理、投资组合优化、量化分析等多种场景。

## 项目演进历程

本项目最初专注于波动率计算，提供了多种协方差矩阵计算方法。随着金融分析需求的不断发展，项目逐步扩展，增加了主成分分析功能、策略评价分析工具和仓位管理工具，形成了更加完整的金融分析工具生态系统。

## 核心模块

### 1. 协方差矩阵计算工具 (Volatility_util.py)

提供多种协方差矩阵计算方法，是风险管理和投资组合优化的基础工具。

### 2. PCA分析器 (PCAanalysis.py)

全功能的主成分分析工具，支持多时间序列的降维分析和可视化。

### 3. 策略评价分析工具 (Eval_util.py)

提供全面的投资策略回测评价指标计算和风险分析功能。

### 4. 仓位管理工具 (Position_util.py)

提供灵活的仓位调整和时间管理功能，支持多种调仓策略。

## 功能特性

### 协方差矩阵计算 (Volatility_util.py)

支持9种不同的协方差矩阵计算方法：

1. **样本协方差 (ALL)** - 传统的样本协方差矩阵计算
2. **半衰协方差 (HALF)** - 基于时间衰减权重的协方差计算
3. **对角协方差 (DIAG)** - 仅保留对角线方差的协方差矩阵
4. **收缩协方差 (SPRING)** - 基于 Ledoit-Wolf 收缩估计的协方差矩阵
5. **阈值协方差 (THRESH)** - 基于阈值过滤的协方差矩阵
6. **EWMA协方差 (EWMA)** - 指数加权移动平均协方差矩阵
7. **GARCH协方差 (GARCH)** - 基于 GARCH 模型的条件协方差矩阵
8. **半协方差 (SEMI)** - 下行风险的半协方差矩阵
9. **EWMA半协方差 (EWMA_SEMI)** - 指数加权移动平均半协方差矩阵

### PCA分析器 (PCAanalysis.py)

提供完整的主成分分析功能：

- **数据预处理**: 自动标准化和数据验证
- **PCA分解**: 灵活的主成分数量设置
- **统计指标**: 方差解释比、累积方差比、特征贡献度等
- **可视化分析**: 方差解释图、成分热力图、双标图等
- **结果导出**: 支持多种格式的结果导出
- **重构误差**: 模型质量评估

### 策略评价分析工具 (Eval_util.py)

提供全面的投资策略回测评价功能：

- **基础指标计算**: 累计收益率、年化收益率、年化波动率、最大回撤
- **风险调整指标**: 夏普比率、卡玛比率等
- **滚动分析**: 滚动累计收益率、滚动回撤分析
- **VaR风险分析**: 历史模拟法计算风险价值(VaR)
- **分年度分析**: 按年度统计各项风险收益指标
- **分月度分析**: 按月度统计各项风险收益指标，包括月度胜率计算
- **投资规模计算**: 基于风险控制的最大投资规模计算
- **结果导出**: 支持Excel格式的详细分析报告

### 仓位管理工具 (Position_util.py)

提供灵活的仓位调整和时间管理功能：

- **多种调仓模式**: 支持固定日期调仓和固定间隔调仓
- **时间序列处理**: 自动处理开仓、平仓时间范围
- **调仓日期生成**: 智能生成调仓时间点
- **数据筛选**: 根据调仓策略筛选相关数据
- **灵活配置**: 支持年度、月度、周度、日度等多种调仓频率

## 依赖库

```python
# 基础数据处理
import pandas as pd
import numpy as np

# 机器学习和统计
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.covariance import LedoitWolf

# 金融建模
from arch import arch_model
from pypfopt.risk_models import semicovariance

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 其他
from scipy import stats
from typing import Optional, Union, List, Tuple, Dict
```

## 安装依赖

```bash
pip install pandas numpy scikit-learn matplotlib seaborn arch pypfopt scipy
```

## 使用方法

### 协方差矩阵计算

```python
import pandas as pd
from Volatility_util import Cov_Matrix

# 准备收益率数据
returns_data = pd.DataFrame(...)  # 您的收益率数据

# 创建协方差矩阵计算实例
cov_calculator = Cov_Matrix(ret=returns_data, method='ALL')

# 计算协方差矩阵
cov_matrix = cov_calculator.calculate_cal_cov_matrix(frequency=252)
```

### PCA分析

```python
import pandas as pd
from PCAanalysis import PCAAnalyzer

# 准备数据
data = pd.DataFrame(...)  # 您的时间序列数据

# 创建PCA分析器
pca_analyzer = PCAAnalyzer(standardize=True, n_components=None)

# 拟合模型
pca_analyzer.fit(data)

# 获取分析结果
explained_variance = pca_analyzer.get_explained_variance_ratio()
components_matrix = pca_analyzer.get_components_matrix()

# 可视化分析
pca_analyzer.plot_explained_variance()
pca_analyzer.plot_components_heatmap()
pca_analyzer.plot_biplot()

# 导出结果
pca_analyzer.export_results('pca_results.xlsx')
```

### 策略评价分析

```python
import pandas as pd
from Eval_util import get_eval_indicator, Year_analysis, Month_analysis

# 准备收益率数据
returns_data = pd.DataFrame(...)  # 您的策略收益率数据

# 计算基础评价指标
eval_indicators = get_eval_indicator(returns_data)
print(eval_indicators)

# 进行年度分析（包含VaR分析）
annual_analysis = Year_analysis(returns_data, dafult_VaR_year_windows=5, save_=True)

# 进行月度分析（包含VaR分析和月度胜率）
monthly_analysis = Month_analysis(returns_data, dafult_VaR_year_windows=5, save_=True)
```

### 仓位管理

```python
import pandas as pd
from Position_util import Position_info

# 准备数据
total_df = pd.DataFrame(...)  # 您的收益率数据
start_date = '2020-01-01'
end_date = '2023-12-31'

# 创建仓位管理实例
position_manager = Position_info(
    total_df=total_df,
    start_date=start_date,
    end_date=end_date,
    change_time_delta=20,  # 每20个交易日调仓
    initial_month=1,
    initial_day=1
)

# 获取调仓信息
position_df, change_position_df, change_dates = position_manager.position_information()
```

### 不同方法示例

```python
# 1. 样本协方差
cov_sample = Cov_Matrix(returns_data, 'ALL')
sample_cov = cov_sample.calculate_cal_cov_matrix()

# 2. EWMA协方差
cov_ewma = Cov_Matrix(returns_data, 'EWMA')
ewma_cov = cov_ewma.calculate_cal_cov_matrix()

# 3. GARCH协方差
cov_garch = Cov_Matrix(returns_data, 'GARCH')
garch_cov = cov_garch.calculate_cal_cov_matrix()

# 4. 半协方差
cov_semi = Cov_Matrix(returns_data, 'SEMI')
semi_cov = cov_semi.calculate_cal_cov_matrix()
```

## 方法详解

### 1. 样本协方差 (ALL)

计算传统的样本协方差矩阵，并进行年化处理。

### 2. 半衰协方差 (HALF)

将数据分为4个子集，对不同时期的协方差赋予不同权重，实现时间衰减效果。

### 3. 对角协方差 (DIAG)

仅保留协方差矩阵的对角线元素（方差），假设资产间无相关性。

### 4. 收缩协方差 (SPRING)

使用 Ledoit-Wolf 收缩估计方法，在样本协方差和结构化估计之间找到最优平衡。

### 5. 阈值协方差 (THRESH)

设置阈值过滤掉较小的协方差值，保留主要的相关性结构。

### 6. EWMA协方差 (EWMA)

使用指数加权移动平均方法，对近期数据赋予更高权重。

### 7. GARCH协方差 (GARCH)

基于 GARCH(1,1) 模型计算条件协方差矩阵，捕捉波动率聚集效应。

### 8. 半协方差 (SEMI)

仅考虑下行风险的协方差计算，适用于风险厌恶的投资场景。

### 9. EWMA半协方差 (EWMA_SEMI)

结合 EWMA 和半协方差的优点，对下行风险进行时间加权计算。

## 参数说明

- `ret`: 收益率数据 (pandas DataFrame)
- `method`: 协方差计算方法 (字符串)
- `frequency`: 年化频率，默认为252 (交易日)
- `lambda_`: EWMA方法中的衰减因子，默认为0.84
- `threshold`: 阈值方法中的过滤阈值，默认为0.03

## 注意事项

1. 输入数据应为适当格式的 pandas DataFrame
2. 确保数据质量，处理缺失值和异常值
3. GARCH 方法计算时间较长，适用于中小规模数据集
4. PCA分析前建议进行数据标准化
5. 根据具体应用场景选择合适的协方差计算方法

## 贡献指南

欢迎提交问题报告、功能请求或代码贡献。请确保：
- 代码符合项目风格
- 添加适当的文档和注释
- 包含必要的测试用例

## 许可证

本项目采用开源许可证，详情请参考项目根目录下的LICENSE文件。

## 应用场景

### 风险管理
- 投资组合风险评估
- 压力测试和情景分析
- 风险因子识别

### 投资组合优化
- 资产配置优化
- 风险平价策略
- 因子投资

### 量化分析
- 多因子模型构建
- 降维分析
- 数据探索性分析
- 策略回测评价
- 投资绩效分析

### 金融建模
- 衍生品定价
- 风险模型构建
- 市场微观结构分析
- 仓位管理优化

## 项目结构

```
Util_Fin/
├── PCAanalysis.py      # PCA分析器主模块
├── Volatility_util.py  # 协方差矩阵计算工具
├── Eval_util.py        # 策略评价分析工具
├── Position_util.py    # 仓位管理工具
├── README.md          # 中文说明文档
└── README_EN.md       # 英文说明文档
```

## 版本历史

- **v1.0**: 基础波动率计算工具
- **v2.0**: 增加多种协方差矩阵计算方法
- **v3.0**: 新增PCA分析功能，形成完整的金融分析工具集
- **v4.0**: 新增策略评价分析和仓位管理功能，构建全面的量化分析平台


---

*更多详细信息请参考英文版 README: [README_EN.md](README_EN.md)*