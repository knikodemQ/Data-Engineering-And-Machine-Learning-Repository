# Ensemble Methods Analysis

This project explores various ensemble learning techniques for classification tasks, demonstrating how combining multiple models can lead to improved predictive performance and robustness compared to individual models.

## Overview

Ensemble methods are machine learning approaches that combine multiple base models to produce one optimal predictive model. These techniques often achieve better results than any single model alone by leveraging the strengths of different models and reducing their individual weaknesses.

## Project Structure

- `ensemble_methods_analysis.ipynb`: Main notebook with comprehensive analysis
- `data/`: Directory containing model artifacts and visualizations
  - `voting_comparison.png`: Performance comparison of voting classifiers
  - `voting_decision_boundaries.png`: Decision boundaries visualization
  - `bagging_comparison.png`: Performance comparison of bagging methods
  - `feature_frequency.png`: Analysis of feature importance in ensemble models
  - `best_models_comparison.png`: Comparison of best models from each method
  - `roc_curves.png`: ROC curve analysis of ensemble performance
  - `cross_validation.png`: Cross-validation results visualization
  - Model serialization files (`.pkl`): Saved model and results files

## Key Features

1. **Voting Classifiers**
   - Implementation of hard and soft voting strategies
   - Comparison with individual base classifiers (Decision Tree, Logistic Regression, KNN)
   - Decision boundary visualization

2. **Bagging Methods**
   - Standard bagging with different sample sizes
   - Pasting (bagging without replacement)
   - Random Forests
   - Feature importance analysis

3. **Boosting Algorithms**
   - AdaBoost implementation
   - Gradient Boosting implementation
   - Performance comparison with bagging methods

4. **Feature Bagging**
   - Training models on different feature subsets
   - Analysis of which features contribute most to performance
   - Ranking estimators by performance

5. **Advanced Analysis**
   - ROC curve and AUC comparison
   - Cross-validation to assess model stability
   - Ensemble method comparison

## Key Concepts Demonstrated

### Voting Classifiers

Voting combines predictions from multiple models:

- **Hard Voting**: Uses majority vote rule (most frequent class prediction)
- **Soft Voting**: Uses weighted average of predicted probabilities

```
Model A: [0.7, 0.3] → Class 0
Model B: [0.4, 0.6] → Class 1 
Model C: [0.3, 0.7] → Class 1

Hard Voting: Class 1 (2 votes vs 1)
Soft Voting: [0.47, 0.53] → Class 1
```

### Bagging (Bootstrap Aggregating)

Bagging trains models on random subsets of the data:

1. Create multiple datasets by sampling with replacement
2. Train a model on each dataset
3. Combine predictions (average for regression, majority vote for classification)

Bagging reduces variance and helps prevent overfitting.

### Boosting

Boosting builds models sequentially:

1. Train initial model
2. Identify misclassified instances
3. Train next model with higher emphasis on previously misclassified instances
4. Continue process, giving higher weight to difficult examples
5. Combine models with weighted voting

Boosting reduces both bias and variance.

### Random Forests

Random Forests combine bagging with random feature selection:

1. Create bootstrap samples
2. For each sample, build a decision tree using only a random subset of features
3. Combine predictions from all trees

This approach decorrelates trees and improves ensemble performance.

## Technologies Used

- Python 3
- NumPy and Pandas for data manipulation
- Scikit-learn for machine learning models
- Matplotlib and Seaborn for data visualization

## Results

The project demonstrates that:

1. Ensemble methods consistently outperform individual models on the breast cancer classification task.
2. Soft voting generally provides better results than hard voting.
3. Random Forest and Gradient Boosting typically achieve the highest accuracy.
4. Feature bagging helps identify which features contribute most to classification performance.
5. Cross-validation shows that ensemble methods also provide more stable results across different data splits.

## Future Work

- Implement stacking (meta-learning) to further improve performance
- Explore more sophisticated ensemble weighting strategies
- Apply these methods to multiclass classification problems
- Investigate computational efficiency and model complexity trade-offs

## References

- Breiman, L. (1996). Bagging predictors. Machine Learning, 24(2), 123-140.
- Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of Computer and System Sciences, 55(1), 119-139.
- Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189-1232.
- Scikit-learn documentation: [Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html)
