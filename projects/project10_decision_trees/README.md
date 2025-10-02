# Decision Trees Analysis

This project explores the application of Decision Trees for both classification and regression tasks. Decision trees are versatile machine learning algorithms that can model non-linear relationships in data by recursively partitioning the feature space.

## Overview

Decision Trees are intuitive, interpretable models that recursively split data into subsets based on feature values to maximize information gain or minimize impurity. They're effective for both classification and regression tasks and can capture non-linear relationships without requiring feature scaling.

## Project Structure

- `decision_trees_analysis.ipynb`: Main notebook with comprehensive analysis
- `data/`: Directory containing model artifacts and visualizations
  - `breast_cancer_tree.png`: Visualization of the decision tree for cancer classification
  - `decision_boundaries.png`: Decision boundaries plot
  - `confusion_matrix.png`: Confusion matrix visualization
  - `feature_importance.png`: Bar chart of feature importance
  - `regression_fit.png`: Visualization of decision tree regression fit
  - `regression_mse_depth.png`: MSE vs depth plot for regression models
  - `pruning_comparison.png`: Comparison of different pruning strategies
  - `categorical_tree.png`: Decision tree with categorical features
  - `f1acc_tree.pkl`: Pickle file with best classification model metrics
  - `mse_tree.pkl`: Pickle file with best regression model metrics

## Key Features

1. **Classification Analysis**
   - Binary classification on the Breast Cancer Wisconsin dataset
   - Comparison of model performance across different tree depths
   - Decision boundary visualization
   - Confusion matrix and classification metrics

2. **Regression Analysis**
   - Fitting polynomial data using decision tree regression
   - Evaluating MSE at different tree depths
   - Comparison with true polynomial function

3. **Advanced Techniques**
   - Feature importance analysis
   - Pruning strategies to control overfitting
   - Handling categorical features

## Key Concepts Demonstrated

### Decision Trees for Classification

Decision trees recursively split the feature space to create pure regions for classification:


### Decision Trees for Regression

For regression, decision trees predict the average target value in leaf nodes:


### Controlling Overfitting

Several techniques are explored to prevent overfitting:
- Setting maximum tree depth
- Requiring minimum samples per split
- Requiring minimum samples per leaf
- Pruning strategies comparison

## Technologies Used

- Python 3
- NumPy and Pandas for data manipulation
- Scikit-learn for machine learning models
- Matplotlib and Seaborn for data visualization

## Results

The project demonstrates that:

1. For the Breast Cancer dataset:
   - Decision trees can achieve high accuracy with proper depth selection
   - Feature importance analysis reveals the most predictive features

2. For regression tasks:
   - Decision trees can approximate complex non-linear functions
   - Tree depth critically impacts model performance

3. General findings:
   - Proper pruning significantly reduces overfitting
   - Decision trees handle categorical features effectively with one-hot encoding
   - Visualizing trees provides interpretability advantage over black-box models

## Future Work

- Implement cost-complexity pruning
- Explore feature selection methods to improve model performance
- Compare with ensemble methods like Random Forests and Gradient Boosting

## References

- Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). Classification and Regression Trees.
- Scikit-learn documentation: [Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- UCI Machine Learning Repository: [Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
