#  Data Engineering & Machine Learning Repository

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-green.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository is a comprehensive collection of data engineering and machine learning projects covering everything from basic data manipulation to advanced deep learning techniques. 

##  Project Core

This repository contains **15 comprehensive projects** covering the full spectrum of data science and machine learning:

- **Data Fundamentals** - Pandas operations, data processing, and cleaning techniques
- **Spatial & Time Series Analysis** - Geographic data and temporal modeling
- **Machine Learning Models** - Classification, regression, and clustering algorithms
- **Advanced ML Techniques** - Ensemble methods, dimensionality reduction, and deep learning
- **Model Optimization** - Hyperparameter tuning and performance optimization

Each project comes with real datasets, well-documented Jupyter notebooks, and practical examples you can run and modify.

##  Repository Structure

```
data-science-repository/
├── projects/                   # 15 comprehensive data science projects
│   ├── project01_pandas_basics/          # Data manipulation fundamentals
│   ├── project02_data_processing/        # ETL pipelines and transformations
│   ├── project03_data_cleaning/         # Data quality and cleaning
│   ├── project04_spatial_data/          # Geographic data analysis
│   ├── project05_time_series/           # Time series analysis and forecasting
│   ├── project06_housing_analysis/      # Real estate market analysis
│   ├── project07_mnist_classification/  # Image classification with neural networks
│   ├── project08_regression_analysis/   # Linear and polynomial regression
│   ├── project09_svm_classification/    # Support Vector Machines
│   ├── project10_decision_trees/        # Decision trees and model interpretation
│   ├── project11_ensemble_methods/      # Random forests, boosting, bagging
│   ├── project12_clustering_analysis/   # K-means, DBSCAN, hierarchical clustering
│   ├── project13_dimensionality_reduction/ # PCA, t-SNE, feature selection
│   ├── project14_deep_learning/         # Neural networks with TensorFlow
│   └── project15_hyperparameter_tuning/ # Model optimization techniques
├── src/                       # Reusable utility modules
│   ├── data/                  # Data loading and preprocessing utilities
│   ├── features/              # Feature engineering functions
│   ├── models/                # Model training and evaluation tools
│   └── visualization/         # Plotting and visualization helpers
├── requirements.txt           # Python dependencies
├── setup.py                  # Package configuration
└── README.md                 # This documentation
```

##  Project Overview

### Data Fundamentals (Projects 1-5)
- **[Project 01: Pandas Basics](projects/project01_pandas_basics/)** - Data manipulation, exploration, and statistical analysis
- **[Project 02: Data Processing](projects/project02_data_processing/)** - ETL pipelines, custom transformations, and data formatting
- **[Project 03: Data Cleaning](projects/project03_data_cleaning/)** - Handling missing data, outliers, and quality assurance
- **[Project 04: Spatial Data Analysis](projects/project04_spatial_data/)** - Geographic data processing and mapping
- **[Project 05: Time Series Analysis](projects/project05_time_series/)** - Temporal data modeling and forecasting

### Machine Learning Foundations (Projects 6-10)  
- **[Project 06: Housing Market Analysis](projects/project06_housing_analysis/)** - Real estate price prediction and market analysis
- **[Project 07: MNIST Classification](projects/project07_mnist_classification/)** - Handwritten digit recognition with neural networks
- **[Project 08: Regression Analysis](projects/project08_regression_analysis/)** - Linear, polynomial, and regularized regression techniques
- **[Project 09: SVM Classification](projects/project09_svm_classification/)** - Support Vector Machines for classification tasks
- **[Project 10: Decision Trees](projects/project10_decision_trees/)** - Decision trees and model interpretation techniques

### Advanced Machine Learning (Projects 11-15)
- **[Project 11: Ensemble Methods](projects/project11_ensemble_methods/)** - Random forests, boosting, and voting classifiers
- **[Project 12: Clustering Analysis](projects/project12_clustering_analysis/)** - K-means, DBSCAN, and hierarchical clustering
- **[Project 13: Dimensionality Reduction](projects/project13_dimensionality_reduction/)** - PCA, t-SNE, and feature selection
- **[Project 14: Deep Learning](projects/project14_deep_learning/)** - Neural networks with TensorFlow and Keras
- **[Project 15: Hyperparameter Tuning](projects/project15_hyperparameter_tuning/)** - Model optimization and performance tuning

##  Technical Stack

### Core Technologies
- **Python 3.9+** - Main programming language
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Matplotlib & Seaborn** - Data visualization and statistical plots
- **Scikit-learn** - Machine learning algorithms and model evaluation
- **TensorFlow & Keras** - Deep learning and neural networks
- **Jupyter Notebooks** - Interactive development and documentation

### Specialized Libraries
- **GeoPandas** - Geographic data analysis and mapping
- **Statsmodels** - Statistical modeling and time series analysis
- **DBSCAN, K-means** - Clustering algorithms and unsupervised learning
- **Keras Tuner** - Automated hyperparameter optimization
- **Plotly** - Interactive visualizations and dashboards

### Data Engineering Tools
- **Pickle & JSON** - Data serialization and storage
- **CSV & Excel** - Structured data formats
- **Custom ETL pipelines** - Data transformation workflows
- **Statistical validation** - Data quality assurance

##  Getting Started

### 1. Clone the Repository
```bash
git clone <repository-url>
cd data-science-repository
```

### 2. Set Up Environment

#### Option A: Using pip
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using conda
```bash
conda create -n datascience python=3.9
conda activate datascience
pip install -r requirements.txt
```

### 3. Launch Jupyter
```bash
jupyter notebook
# or
jupyter lab
```

### 4. Start Exploring
Open any project folder and launch the `.ipynb` notebook to start learning!

##  Skills & Techniques Covered

### Data Engineering
- **Data Processing Pipelines** - ETL workflows and data transformations
- **Data Cleaning** - Handling missing values, outliers, and inconsistencies
- **Feature Engineering** - Creating and selecting meaningful features
- **Data Validation** - Quality checks and automated testing

### Machine Learning
- **Supervised Learning** - Classification and regression algorithms
- **Unsupervised Learning** - Clustering and dimensionality reduction
- **Deep Learning** - Neural networks with TensorFlow and Keras
- **Model Evaluation** - Cross-validation, metrics, and performance analysis

### Advanced Techniques
- **Ensemble Methods** - Random forests, boosting, and model combination
- **Hyperparameter Tuning** - Grid search, random search, and Bayesian optimization
- **Time Series Analysis** - Forecasting and temporal pattern detection
- **Spatial Analysis** - Geographic data processing and mapping

### Tools & Frameworks
- **Data Manipulation** - Pandas for data wrangling and analysis
- **Visualization** - Matplotlib, Seaborn, and Plotly for insights
- **ML Libraries** - Scikit-learn for traditional ML, TensorFlow for deep learning
- **Optimization** - Keras Tuner and various hyperparameter search strategies

### Beginners (Projects 1-5)
Start with the fundamentals to build a solid foundation:
1. **Project 01-03** - Master data manipulation, processing, and cleaning
2. **Project 04-05** - Learn spatial and time series analysis
3. **Explore src/ modules** - Understand reusable code patterns

### Intermediate (Projects 6-10)
Dive into machine learning basics:
1. **Project 06-08** - Housing analysis, MNIST, and regression techniques
2. **Project 09-10** - SVM classification and decision trees
3. **Focus on model evaluation** - Learn proper validation techniques

### Advanced (Projects 11-15)
Master advanced ML techniques:
1. **Project 11-13** - Ensemble methods, clustering, and dimensionality reduction
2. **Project 14-15** - Deep learning and hyperparameter optimization
3. **Integration projects** - Combine techniques across projects

### Comprehensive Coverage
- **Full ML pipeline** from data engineering to model deployment
- **Multiple domains** including time series, spatial data, and deep learning  
- **Various techniques** from basic statistics to advanced neural networks

##  The src/ Directory

The `src/` folder contains reusable utility modules that can be use to organize code:

- **`data/`** - Data loading, preprocessing, and I/O utilities
- **`features/`** - Feature engineering and transformation functions  
- **`models/`** - Model training, evaluation, and utility functions
- **`visualization/`** - Plotting helpers and visualization tools

Repository provides:

- **Hands-on experience** with real data science workflows
- **Progressive learning** from basics to advanced techniques  
- **Professional practices** for organizing and documenting code
- **Diverse applications** across multiple domains and use cases
- **Complete examples** you can modify for your own projects

