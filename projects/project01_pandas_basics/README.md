# Project 01: Pandas Basics - Data Analysis and Manipulation

##  Overview

This project demonstrates fundamental pandas operations for data analysis and manipulation. It covers essential techniques for data scientists and analysts working with structured data in Python.

##  Learning Objectives

- Master basic pandas data loading and exploration techniques
- Understand data types and column analysis methods
- Generate comprehensive statistical summaries
- Learn data cleaning and column normalization strategies
- Practice multi-format data export and import operations
- Work with different data formats (CSV, JSON, Pickle, Excel)

##  Project Structure

```
project01_pandas_basics/
├── data/                              # Input datasets
│   ├── proj1_ex01.csv                # Main dataset for analysis
│   ├── proj1_ex05.pkl                # Pickle format dataset
│   └── proj1_ex06.json               # JSON format dataset
├── output/                           # Generated output files
├── pandas_basics_analysis.ipynb     # Main analysis notebook
└── README.md                        # This file
```

##  Dataset Description

### Main Dataset (proj1_ex01.csv)
- **Purpose**: Demonstration of basic pandas operations
- **Format**: CSV (Comma Separated Values)
- **Content**: Sample structured data with mixed data types
- **Use Case**: Practice data loading, exploration, and basic analysis

### Additional Datasets
- **proj1_ex05.pkl**: Pickle format data for advanced selection operations
- **proj1_ex06.json**: JSON format data for normalization exercises

##  Technical Skills Demonstrated

### 1. Data Loading and Exploration
- df.info()
- df.head()
- df.describe()

### 2. Column Analysis and Metadata Generation
- Automatic data type detection
- Missing value analysis
- Unique value counting
- Memory usage optimization

### 3. Statistical Summary Generation
- Numerical columns: mean, std, min, max, quantiles
- Categorical columns: unique counts, frequency analysis
- Missing data handling

### 4. Data Cleaning Techniques
```python
# Column name normalization
cleaned_columns = [re.sub(r"[^A-Za-z0-9_ ]", "", col)
                  .lower().replace(" ", "_").strip() 
                  for col in df.columns]
```

### 5. Multi-Format Data Export
- **CSV**: Standard tabular format
- **JSON**: Records and normalized formats
- **Pickle**: Python-specific serialization
- **Excel**: Business-friendly format
- **Parquet**: Efficient columnar storage

##  Key Functions Implemented

### `analyze_columns(dataframe)`
Analyzes each column and returns comprehensive metadata including:
- Data type detection
- Missing value percentages
- Unique value counts
- Memory usage statistics

### `generate_statistical_summary(dataframe)`
Creates detailed statistical summaries:
- Numerical: descriptive statistics
- Categorical: frequency analysis
- Missing data reporting

### `clean_column_names(dataframe)`
Normalizes column names by:
- Removing special characters
- Converting to lowercase
- Replacing spaces with underscores
- Handling multiple consecutive underscores

