# Dimensionality Reduction with Principal Component Analysis (PCA)

This project demonstrates Principal Component Analysis (PCA), a fundamental dimensionality reduction technique, applied to two classic machine learning datasets: Breast Cancer Wisconsin and Iris.

## Overview

PCA is an unsupervised learning technique that transforms high-dimensional data into a lower-dimensional representation while preserving as much variance as possible. This project explores PCA theory, implementation, and practical applications.

## Project Structure

- `dimensionality_reduction_analysis.ipynb`: Comprehensive PCA analysis notebook
- `data/`: Results, visualizations, and saved models
  - `pca_bc.pkl`, `pca_ir.pkl`: Explained variance ratios (original lab outputs)
  - `idx_bc.pkl`, `idx_ir.pkl`: Most important feature indices (original lab outputs)
  - `pca_model_bc.pkl`, `pca_model_ir.pkl`: Complete fitted PCA models
  - `scaler_bc.pkl`, `scaler_ir.pkl`: StandardScaler objects
  - `data_bc_pca.npy`, `data_ir_pca.npy`: Transformed datasets
  - Visualization files: Various plots showing PCA results and analysis

## Key Features

1. **Data Preprocessing**
   - Feature standardization for PCA sensitivity
   - Exploratory data analysis of both datasets
   - Distribution visualization of original features

2. **PCA Implementation**
   - Variance-based component selection (90% variance retention)
   - Explained variance ratio analysis
   - Component loading interpretation

3. **Feature Analysis**
   - Identification of most important features per component
   - Loading visualization and interpretation
   - Component contribution analysis

4. **Data Visualization**
   - 2D and 3D projections of high-dimensional data
   - Class separation in reduced dimensional space
   - Component variance comparison

5. **Comparative Analysis**
   - Different variance threshold requirements
   - Components needed vs. variance explained
   - Efficiency of dimensionality reduction

## Datasets Used

### Breast Cancer Wisconsin Dataset
- **Features**: 30 numerical features (tumor characteristics)
- **Samples**: 569 instances
- **Classes**: Malignant (0) and Benign (1)
- **PCA Result**: Significant dimensionality reduction while maintaining class separability

### Iris Dataset
- **Features**: 4 numerical features (flower measurements)
- **Samples**: 150 instances  
- **Classes**: 3 iris species (setosa, versicolor, virginica)
- **PCA Result**: Nearly perfect variance capture with minimal components

## Key Concepts

### Principal Component Analysis
- **Objective**: Find directions of maximum variance in data
- **Method**: Eigenvalue decomposition of covariance matrix
- **Output**: Orthogonal components ordered by explained variance

### Dimensionality Reduction Benefits
- Reduced computational complexity
- Visualization of high-dimensional data
- Noise reduction through variance filtering
- Storage efficiency improvement

### Component Interpretation
- **Loading Values**: Contribution of each original feature to a component
- **Explained Variance**: Proportion of total variance captured by each component
- **Cumulative Variance**: Total variance explained by first n components

## Implementation Details

```python
# Core PCA implementation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(original_data)

# Apply PCA with 90% variance retention
pca = PCA(n_components=0.90)
data_pca = pca.fit_transform(data_scaled)

# Analyze results
explained_variance = pca.explained_variance_ratio_
important_features = [np.argmax(np.abs(component)) for component in pca.components_]
```

## Results

### Breast Cancer Dataset
- **Original Dimensions**: 30 features
- **Reduced Dimensions**: ~6-8 components (90% variance)
- **First PC**: Captures majority of variance
- **Class Separation**: Clearly visible in PCA space

### Iris Dataset  
- **Original Dimensions**: 4 features
- **Reduced Dimensions**: 2-3 components (90% variance)
- **First PC**: Primarily captures petal measurements
- **Species Separation**: Excellent separation in 2D PCA space

## Technologies Used

- **Python 3**: Core programming language
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: PCA implementation and preprocessing
- **Matplotlib/Seaborn**: Data visualization and plotting

## Applications

1. **Preprocessing for ML**: Reduce dimensionality before applying other algorithms
2. **Data Visualization**: Enable plotting of high-dimensional data
3. **Feature Engineering**: Create meaningful composite features
4. **Anomaly Detection**: Identify outliers using reconstruction error
5. **Data Compression**: Store data more efficiently

## Advantages and Limitations

### Advantages
- Mathematically principled approach
- Preserves maximum variance
- Provides interpretable components
- Computational efficiency
- No labeled data required

### Limitations
- Linear transformation only
- Components may not be interpretable
- Sensitive to feature scaling
- May not preserve local structure
- Requires full dataset for fitting

## Future Extensions

- Compare with other dimensionality reduction techniques (t-SNE, UMAP)
- Implement kernel PCA for non-linear relationships
- Apply PCA for image compression and reconstruction
- Use PCA components as features in supervised learning
- Explore incremental PCA for large datasets
- Investigate sparse PCA for feature selection

## References

- Jolliffe, I. T. (2002). Principal Component Analysis. Springer.
- Scikit-learn documentation: [PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- Shlens, J. (2014). A tutorial on principal component analysis. arXiv preprint arXiv:1404.1100.
- Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine.
