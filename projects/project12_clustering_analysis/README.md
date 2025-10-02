# Clustering Analysis

This project explores unsupervised learning techniques through clustering algorithms, focusing on K-Means and DBSCAN methods applied to the MNIST handwritten digits dataset.

## Overview

Clustering is an unsupervised machine learning approach that groups similar data points together based on their intrinsic properties without using labeled data for training. This project demonstrates how different clustering algorithms can discover patterns in high-dimensional data.

## Project Structure

- `clustering_analysis.ipynb`: Main notebook with comprehensive analysis
- `data/`: Directory containing model artifacts and visualizations
  - `sample_digits.png`: Sample MNIST digits visualization
  - `mnist_pca.png`: PCA-reduced MNIST data visualization
  - `mnist_tsne.png`: t-SNE visualization of MNIST data
  - `kmeans_silhouette_scores.png`: Silhouette scores for different K values
  - `kmeans_clusters_pca.png`: Visualization of K-means clusters
  - `kmeans_confusion_matrix.png`: Confusion matrix between true digits and clusters
  - `kmeans_centroids.png`: Visualization of cluster centers as digit images
  - `cluster_*_samples.png`: Sample images from each cluster
  - `dbscan_k_distance.png`: K-distance graph for DBSCAN parameter selection
  - `dbscan_parameters.png`: Analysis of DBSCAN parameters
  - `dbscan_clusters_pca.png`: Visualization of DBSCAN clusters
  - `dbscan_confusion_matrix.png`: Confusion matrix for DBSCAN results
  - `dbscan_noise_samples.png`: Visualization of noise points
  - `dbscan_noise_by_digit.png`: Analysis of noise points by digit class
  - `hierarchical_clusters.png`: Hierarchical clustering visualization
  - `anomaly_digits.png`: Unusual digits detected as anomalies
  - Serialized model files (`.pkl`): Saved model parameters and results

## Key Features

1. **Dataset Preparation**
   - Loading and exploring the MNIST handwritten digits dataset
   - Dimensionality reduction using PCA and t-SNE for visualization
   - Sampling strategies for computational efficiency

2. **K-Means Clustering**
   - Algorithm overview and implementation
   - Finding optimal K using silhouette scores
   - Visualizing cluster centroids as digit prototypes
   - Evaluation using confusion matrix and adjusted Rand index

3. **DBSCAN Clustering**
   - Density-based clustering approach
   - Parameter tuning (eps and min_samples)
   - Outlier detection capabilities
   - Performance evaluation and cluster visualization

4. **Comparative Analysis**
   - Quantitative comparison of clustering methods
   - Strengths and weaknesses of each approach
   - Performance metrics including silhouette score and adjusted Rand index

5. **Advanced Applications**
   - Hierarchical clustering for additional perspective
   - Anomaly detection using strict DBSCAN parameters
   - Insights into cluster interpretability

## Key Concepts Demonstrated

### K-Means Clustering

K-Means partitions data into K clusters by:
1. Initializing K cluster centroids randomly
2. Assigning each data point to the nearest centroid
3. Recalculating centroids as the mean of assigned points
4. Iterating until convergence

```
Optimize: Sum of squared distances from points to their cluster centers
```

### DBSCAN Clustering

DBSCAN forms clusters based on density:
1. Core points: Have at least min_samples points within distance eps
2. Border points: Within eps of core points but have fewer neighbors
3. Noise points: Neither core nor border points

```
Parameters:
- eps: Maximum distance between two points to be considered neighbors
- min_samples: Minimum points required to form a dense region
```

### Dimensionality Reduction

High-dimensional data visualization using:
- PCA: Linear technique that preserves maximum variance
- t-SNE: Non-linear technique that preserves local structure

### Cluster Evaluation

Methods for evaluating clustering quality:
- Silhouette score: Measures how similar points are to their own cluster versus other clusters
- Adjusted Rand index: Measures similarity between true labels and cluster assignments
- Confusion matrix: Shows the correspondence between clusters and ground truth

## Technologies Used

- Python 3
- NumPy and Pandas for data manipulation
- Scikit-learn for machine learning models
- Matplotlib and Seaborn for data visualization

## Results

The project demonstrates that:

1. K-means effectively groups MNIST digits into clusters that largely correspond to digit classes, with centroids representing prototypical digit patterns.

2. DBSCAN provides insights through outlier detection, identifying unusual digit samples that might represent writing variations or errors.

3. Dimensionality reduction techniques like PCA and t-SNE are essential for visualizing and understanding high-dimensional data structures.

4. The optimal number of clusters for K-means closely aligns with the natural number of digit classes (10), supporting the algorithm's effectiveness.

5. Parameter selection significantly impacts clustering performance, with systematic approaches (like k-distance graphs) providing better guidance than arbitrary choices.

## Future Work

- Implement additional clustering algorithms (e.g., spectral clustering, Gaussian mixtures)
- Explore semi-supervised approaches using partial labels
- Investigate cluster stability across different initializations and parameters
- Apply clustering for feature engineering in supervised learning tasks
- Extend to other datasets with different characteristics

## References

- Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In KDD (Vol. 96, No. 34, pp. 226-231).
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In Proceedings of the fifth Berkeley symposium on mathematical statistics and probability (Vol. 1, No. 14, pp. 281-297).
- Scikit-learn documentation: [Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- LeCun, Y., Cortes, C., & Burges, C. (2010). MNIST handwritten digit database. AT&T Labs [Online]. Available: http://yann.lecun.com/exdb/mnist
