# Covariance Matrix Calculation Utility (Volatility_util.py)

## Project Overview

This project provides a  covariance matrix calculation utility class `Cov_Matrix` that supports multiple covariance matrix calculation methods, suitable for financial risk management, portfolio optimization, and other scenarios.

## Features

### Supported Covariance Matrix Calculation Methods

1. **Sample Covariance (ALL)** - Traditional sample covariance matrix calculation
2. **Half-life Covariance (HALF)** - Covariance calculation based on time-decay weights
3. **Diagonal Covariance (DIAG)** - Covariance matrix retaining only diagonal variances
4. **Shrinkage Covariance (SPRING)** - Covariance matrix based on Ledoit-Wolf shrinkage estimation
5. **Threshold Covariance (THRESH)** - Threshold-filtered covariance matrix
6. **EWMA Covariance (EWMA)** - Exponentially Weighted Moving Average covariance matrix
7. **GARCH Covariance (GARCH)** - Conditional covariance matrix based on GARCH model
8. **Semi-covariance (SEMI)** - Semi-covariance matrix for downside risk
9. **EWMA Semi-covariance (EWMA_SEMI)** - Exponentially Weighted Moving Average semi-covariance matrix

## Dependencies

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

## Installation

```bash
pip install pandas numpy scikit-learn arch pypfopt scipy
```

## Usage

### Basic Usage

```python
import pandas as pd
from Volatility_util import Cov_Matrix

# Prepare returns data (DataFrame format)
returns_data = pd.DataFrame(...)  # Your returns data

# Create covariance matrix calculator instance
cov_calculator = Cov_Matrix(ret=returns_data, method='ALL')

# Calculate covariance matrix (default annualization frequency is 252)
cov_matrix = cov_calculator.calculate_cal_cov_matrix(frequency=252)
```

### Examples with Different Methods

```python
# 1. Sample Covariance
cov_sample = Cov_Matrix(returns_data, 'ALL')
sample_cov = cov_sample.calculate_cal_cov_matrix()

# 2. EWMA Covariance
cov_ewma = Cov_Matrix(returns_data, 'EWMA')
ewma_cov = cov_ewma.calculate_cal_cov_matrix()

# 3. GARCH Covariance
cov_garch = Cov_Matrix(returns_data, 'GARCH')
garch_cov = cov_garch.calculate_cal_cov_matrix()

# 4. Semi-covariance
cov_semi = Cov_Matrix(returns_data, 'SEMI')
semi_cov = cov_semi.calculate_cal_cov_matrix()
```

## Method Details

### 1. Sample Covariance (ALL)

Calculates traditional sample covariance matrix with annualization.

### 2. Half-life Covariance (HALF)

Divides data into 4 subsets and assigns different weights to different time periods, achieving time decay effects.

### 3. Diagonal Covariance (DIAG)

Retains only diagonal elements (variances) of the covariance matrix, assuming no correlation between assets.

### 4. Shrinkage Covariance (SPRING)

Uses Ledoit-Wolf shrinkage estimation method to find optimal balance between sample covariance and structured estimation.

### 5. Threshold Covariance (THRESH)

Sets threshold to filter out smaller covariance values, retaining main correlation structure.

### 6. EWMA Covariance (EWMA)

Uses exponentially weighted moving average method, assigning higher weights to recent data.

### 7. GARCH Covariance (GARCH)

Calculates conditional covariance matrix based on GARCH(1,1) model, capturing volatility clustering effects.

### 8. Semi-covariance (SEMI)

Considers only downside risk covariance calculation, suitable for risk-averse investment scenarios.

### 9. EWMA Semi-covariance (EWMA_SEMI)

Combines advantages of EWMA and semi-covariance, performing time-weighted calculation for downside risk.

## Parameters

- `ret`: Returns data (pandas DataFrame)
- `method`: Covariance calculation method (string)
- `frequency`: Annualization frequency, default is 252 (trading days)
- `lambda_`: Decay factor in EWMA method, default is 0.84
- `threshold`: Filter threshold in threshold method, default is 0.03

## Important Notes

1. Input data should be returns format pandas DataFrame
2. Ensure no missing values in data or handle them appropriately
3. GARCH method is computationally intensive, suitable for smaller datasets
4. Semi-covariance method requires price data rather than returns data

## Application Scenarios

- Portfolio risk management
- Asset allocation optimization
- Risk model construction
- Financial derivatives pricing
- Stress testing and scenario analysis

---

*For more detailed information, please refer to the Chinese version README: [README.md](README.md)*
