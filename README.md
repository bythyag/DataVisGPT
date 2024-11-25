# Data Visualisation GPT 🔍📊

## Overview

The Data Insights Agent is a powerful Python-based tool for automated data preprocessing, analysis, and visualization. It provides a comprehensive workflow for loading, cleaning, analyzing, and reporting on datasets with optional AI-enhanced preprocessing.

## Features

### 🌟 Key Capabilities
- Multi-format data loading (CSV, Excel, JSON)
- Advanced data preprocessing
- Comprehensive data analysis
- Intelligent missing value handling
- Outlier detection and removal
- Automated data type conversion
- Optional AI-driven preprocessing with OpenAI
- Visualization generation
- Detailed reporting

### 🧩 Preprocessing Techniques
- Missing value analysis and strategic handling
- Outlier removal using Z-score method
- Automatic data type conversion
- Optional AI-enhanced preprocessing suggestions

### 📈 Analysis Capabilities
- Summary statistics
- Correlation matrix computation
- Categorical column insights
- Distribution visualizations
- Correlation heatmaps

## Prerequisites

### 🔧 Required Libraries
- pandas
- numpy
- matplotlib
- seaborn
- openai (optional)

### 🐍 Python Version
- Python 3.8+

## Installation

1. Clone the repository
```bash
git clone https://github.com/bythyag/DataVisGPT.git
cd DataVisGPT
```

2. Install required dependencies
```bash
pip install pandas numpy matplotlib seaborn openai
```

### Configuration Options
- `outlier_threshold`: Adjust Z-score threshold for outlier removal
- `missing_value_threshold`: Set percentage of missing values to trigger column dropping

## Advanced Features

### 🤖 AI-Enhanced Preprocessing
- Requires OpenAI API key
- Generates preprocessing recommendations
- Provides feature engineering suggestions

## Outputs

### 📄 Generated Files
- `data_insights_report.md`: Comprehensive markdown report
- `visualizations/`: Directory containing generated plots
  - Individual column distribution plots
  - Correlation heatmap

## License

Distributed under the MIT License. See `LICENSE` for more information.
