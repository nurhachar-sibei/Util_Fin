# Financial Analysis Toolkit (Util_Fin)

## Project Overview

This project is a comprehensive financial analysis toolkit that has evolved from an initial volatility calculation tool. It now encompasses a complete suite of financial analysis capabilities, including covariance matrix calculations and Principal Component Analysis (PCA). The toolkit is designed for financial risk management, portfolio optimization, quantitative analysis, and various other financial applications.

## Evolution History

This project initially focused on volatility calculations, providing multiple covariance matrix calculation methods. As financial analysis requirements continued to evolve, the project gradually expanded to include Principal Component Analysis functionality, forming a more complete financial analysis ecosystem.

## Core Modules

### 1. Covariance Matrix Calculator (Volatility_util.py)

Provides multiple covariance matrix calculation methods, serving as a fundamental tool for risk management and portfolio optimization.

### 2. PCA Analyzer (PCAanalysis.py)

A full-featured Principal Component Analysis tool supporting dimensionality reduction analysis and visualization for multiple time series.

## Features

### Covariance Matrix Calculation (Volatility_util.py)

Supports 9 different covariance matrix calculation methods:

1. **Sample Covariance (ALL)** - Traditional sample covariance matrix calculation
2. **Half-life Covariance (HALF)** - Covariance calculation based on time-decay weights
3. **Diagonal Covariance (DIAG)** - Covariance matrix retaining only diagonal variances
4. **Shrinkage Covariance (SPRING)** - Covariance matrix based on Ledoit-Wolf shrinkage estimation
5. **Threshold Covariance (THRESH)** - Threshold-filtered covariance matrix
6. **EWMA Covariance (EWMA)** - Exponentially Weighted Moving Average covariance matrix
7. **GARCH Covariance (GARCH)** - Conditional covariance matrix based on GARCH model
8. **Semi-covariance (SEMI)** - Semi-covariance matrix for downside risk
9. **EWMA Semi-covariance (EWMA_SEMI)** - Exponentially Weighted Moving Average semi-covariance matrix

### PCA Analyzer (PCAanalysis.py)

Provides comprehensive Principal Component Analysis functionality:

- **Data Preprocessing**: Automatic standardization and data validation
- **PCA Decomposition**: Flexible principal component number settings
- **Statistical Metrics**: Explained variance ratio, cumulative variance ratio, feature contributions, etc.
- **Visualization**: Explained variance plots, component heatmaps, biplots, etc.
- **Results Export**: Support for multiple output formats
- **Reconstruction Error**: Model quality assessment

## Dependencies

```python
# Basic data processing
import pandas as pd
import numpy as np

# Machine learning and statistics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.covariance import LedoitWolf

# Financial modeling
from arch import arch_model
from pypfopt.risk_models import semicovariance

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Others
from scipy import stats
from typing import Optional, Union, List, Tuple, Dict
```

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn arch pypfopt scipy
```

## Usage

### Covariance Matrix Calculation

```python
import pandas as pd
from Volatility_util import Cov_Matrix

# Prepare returns data
returns_data = pd.DataFrame(...)  # Your returns data

# Create covariance matrix calculator instance
cov_calculator = Cov_Matrix(ret=returns_data, method='ALL')

# Calculate covariance matrix
cov_matrix = cov_calculator.calculate_cal_cov_matrix(frequency=252)
```

### PCA Analysis

```python
import pandas as pd
from PCAanalysis import PCAAnalyzer

# Prepare data
data = pd.DataFrame(...)  # Your time series data

# Create PCA analyzer
pca_analyzer = PCAAnalyzer(standardize=True, n_components=None)

# Fit the model
pca_analyzer.fit(data)

# Get analysis results
explained_variance = pca_analyzer.get_explained_variance_ratio()
components_matrix = pca_analyzer.get_components_matrix()

# Visualization
pca_analyzer.plot_explained_variance()
pca_analyzer.plot_components_heatmap()
pca_analyzer.plot_biplot()

# Export results
pca_analyzer.export_results('pca_results.xlsx')
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

1. Input data should be properly formatted pandas DataFrames
2. Ensure data quality by handling missing values and outliers
3. GARCH method is computationally intensive, suitable for small to medium datasets
4. Data standardization is recommended before PCA analysis
5. Choose appropriate covariance calculation methods based on specific use cases

## Contributing

We welcome issue reports, feature requests, and code contributions. Please ensure:
- Code follows project style guidelines
- Appropriate documentation and comments are included
- Necessary test cases are provided

## License

This project is released under an open-source license. Please refer to the LICENSE file in the project root directory for details.

## Application Scenarios

### Risk Management
- Portfolio risk assessment
- Stress testing and scenario analysis
- Risk factor identification

### Portfolio Optimization
- Asset allocation optimization
- Risk parity strategies
- Factor investing

### Quantitative Analysis
- Multi-factor model construction
- Dimensionality reduction analysis
- Exploratory data analysis

### Financial Modeling
- Derivatives pricing
- Risk model construction
- Market microstructure analysis

## Project Structure

```
Util_Fin/
├── PCAanalysis.py      # PCA analyzer main module
├── Volatility_util.py  # Covariance matrix calculation utility
├── README.md          # Chinese documentation
└── README_EN.md       # English documentation
```

## Version History

- **v1.0**: Basic volatility calculation tool
- **v2.0**: Added multiple covariance matrix calculation methods
- **v3.0**: Added PCA analysis functionality, forming a complete financial analysis toolkit

---

*For more detailed information, please refer to the Chinese version README: [README.md](README.md)*