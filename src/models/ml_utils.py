import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelEvaluator:
    
    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                             average: str = 'weighted') -> Dict[str, float]:
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
    
    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    @staticmethod
    def cross_validation_scores(model, X: np.ndarray, y: np.ndarray, 
                              cv: int = 5, scoring: str = 'accuracy') -> Dict[str, float]:
        from sklearn.model_selection import cross_val_score
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }

class FeatureSelector:
    
    @staticmethod
    def correlation_filter(df: pd.DataFrame, target_col: str, 
                          threshold: float = 0.1) -> List[str]:
        correlations = df.corr()[target_col].abs()
        return correlations[correlations > threshold].index.tolist()
    
    @staticmethod
    def variance_filter(df: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        variances = df.var()
        return variances[variances > threshold].index.tolist()
    
    @staticmethod
    def mutual_info_filter(X: pd.DataFrame, y: pd.Series, 
                          k: int = 10, task_type: str = 'classification') -> List[str]:
        if task_type == 'classification':
            from sklearn.feature_selection import mutual_info_classif as mutual_info
        else:
            from sklearn.feature_selection import mutual_info_regression as mutual_info
        
        scores = mutual_info(X, y)
        feature_scores = pd.Series(scores, index=X.columns)
        return feature_scores.nlargest(k).index.tolist()

class ModelTrainer:
    
    @staticmethod
    def train_test_split_advanced(df: pd.DataFrame, target_col: str, 
                                test_size: float = 0.2, 
                                stratify: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        from sklearn.model_selection import train_test_split
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        stratify_param = y if stratify else None
        
        return train_test_split(X, y, test_size=test_size, 
                              stratify=stratify_param, random_state=42)
    
    @staticmethod
    def hyperparameter_tuning(model, param_grid: Dict, X: np.ndarray, y: np.ndarray, 
                            cv: int = 5, scoring: str = 'accuracy') -> Any:
        from sklearn.model_selection import GridSearchCV
        
        grid_search = GridSearchCV(model, param_grid, cv=cv, 
                                 scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        
        return grid_search
