import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple

class DataVisualizer:
    
    @staticmethod
    def setup_style(style: str = 'whitegrid', palette: str = 'husl', 
                   figsize: Tuple[int, int] = (12, 8)) -> None:
        sns.set_style(style)
        sns.set_palette(palette)
        plt.rcParams['figure.figsize'] = figsize
    
    @staticmethod
    def correlation_heatmap(df: pd.DataFrame, figsize: Tuple[int, int] = (10, 8), 
                           annot: bool = True, cmap: str = 'coolwarm') -> None:
        plt.figure(figsize=figsize)
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, center=0, 
                   square=True, fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def distribution_plots(df: pd.DataFrame, columns: List[str], 
                          ncols: int = 3, figsize: Tuple[int, int] = (15, 10)) -> None:
        nrows = (len(columns) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        if nrows == 1:
            axes = [axes] if ncols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            sns.histplot(data=df, x=col, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
        
        for i in range(len(columns), len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def scatter_matrix(df: pd.DataFrame, columns: List[str], 
                      hue: Optional[str] = None, figsize: Tuple[int, int] = (12, 10)) -> None:
        if hue:
            sns.pairplot(df[columns + [hue]], hue=hue, diag_kind='hist')
        else:
            sns.pairplot(df[columns], diag_kind='hist')
        plt.show()
    
    @staticmethod
    def time_series_plot(df: pd.DataFrame, x_col: str, y_cols: List[str], 
                        figsize: Tuple[int, int] = (15, 8)) -> None:
        plt.figure(figsize=figsize)
        
        for col in y_cols:
            plt.plot(df[x_col], df[col], label=col, linewidth=2)
        
        plt.xlabel(x_col)
        plt.ylabel('Values')
        plt.title('Time Series Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def box_plots(df: pd.DataFrame, columns: List[str], 
                 ncols: int = 3, figsize: Tuple[int, int] = (15, 10)) -> None:
        nrows = (len(columns) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        if nrows == 1:
            axes = [axes] if ncols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            sns.boxplot(data=df, y=col, ax=axes[i])
            axes[i].set_title(f'Box Plot of {col}')
        
        for i in range(len(columns), len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        plt.show()

class StatisticalPlotter:
    
    @staticmethod
    def qq_plot(data: pd.Series, distribution: str = 'norm') -> None:
        from scipy import stats
        
        fig, ax = plt.subplots(figsize=(8, 6))
        stats.probplot(data, dist=distribution, plot=ax)
        ax.set_title(f'Q-Q Plot ({distribution} distribution)')
        plt.show()
    
    @staticmethod
    def residual_plots(y_true: np.ndarray, y_pred: np.ndarray, 
                      figsize: Tuple[int, int] = (12, 5)) -> None:
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='red', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        
        sns.histplot(residuals, kde=True, ax=axes[1])
        axes[1].set_title('Distribution of Residuals')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def feature_importance_plot(feature_names: List[str], importances: np.ndarray, 
                              top_n: int = 20, figsize: Tuple[int, int] = (10, 8)) -> None:
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=figsize)
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45)
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.show()
