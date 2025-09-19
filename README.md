# 协方差矩阵计算工具 (Volatility_util.py)

## 项目简介

本项目提供了一个协方差矩阵计算工具类 `Cov_Matrix`，支持多种协方差矩阵计算方法，适用于金融风险管理、投资组合优化等场景。

## 功能特性

### 支持的协方差矩阵计算方法

1. **样本协方差 (ALL)** - 传统的样本协方差矩阵计算
2. **半衰协方差 (HALF)** - 基于时间衰减权重的协方差计算
3. **对角协方差 (DIAG)** - 仅保留对角线方差的协方差矩阵
4. **收缩协方差 (SPRING)** - 基于 Ledoit-Wolf 收缩估计的协方差矩阵
5. **阈值协方差 (THRESH)** - 基于阈值过滤的协方差矩阵
6. **EWMA协方差 (EWMA)** - 指数加权移动平均协方差矩阵
7. **GARCH协方差 (GARCH)** - 基于 GARCH 模型的条件协方差矩阵
8. **半协方差 (SEMI)** - 下行风险的半协方差矩阵
9. **EWMA半协方差 (EWMA_SEMI)** - 指数加权移动平均半协方差矩阵

## 依赖库

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from arch import arch_model
from pypfopt.risk_models import semicovariance
from sklearn.covariance import LedoitWolf
from scipy import stats
```

## 安装依赖

```bash
pip install pandas numpy scikit-learn arch pypfopt scipy
```

## 使用方法

### 基本用法

```python
import pandas as pd
from Volatility_util import Cov_Matrix

# 准备收益率数据 (DataFrame格式)
returns_data = pd.DataFrame(...)  # 您的收益率数据

# 创建协方差矩阵计算实例
cov_calculator = Cov_Matrix(ret=returns_data, method='ALL')

# 计算协方差矩阵 (默认年化频率为252)
cov_matrix = cov_calculator.calculate_cal_cov_matrix(frequency=252)
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

1. 输入数据应为收益率格式的 pandas DataFrame
2. 确保数据中无缺失值或已妥善处理
3. GARCH 方法计算时间较长，适用于较小规模的数据集
4. 半协方差方法需要输入价格数据而非收益率数据

## 应用场景

- 投资组合风险管理
- 资产配置优化
- 风险模型构建
- 金融衍生品定价
- 压力测试和情景分析

## 许可证

本项目采用开源许可证，具体请查看 LICENSE 文件。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进本项目。

---

*更多详细信息请参考英文版 README: [README_EN.md](README_EN.md)*
