import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional
import logging

class DataValidator:
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        report = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'null_counts': df.isnull().sum().to_dict(),
            'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        return report
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, columns: List[str], 
                       method: str = 'iqr') -> Dict[str, int]:
        outlier_counts = {}
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = (z_scores > 3).sum()
            
            outlier_counts[col] = outliers
        
        return outlier_counts
    
    @staticmethod
    def validate_data_types(df: pd.DataFrame, expected_types: Dict[str, str]) -> Dict[str, bool]:
        validation_results = {}
        
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                validation_results[col] = expected_type in actual_type
            else:
                validation_results[col] = False
        
        return validation_results

class PerformanceOptimizer:
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        df_optimized = df.copy()
        
        for col in df_optimized.columns:
            col_type = df_optimized[col].dtype
            
            if col_type == 'object':
                try:
                    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
                except (ValueError, TypeError):
                    pass
            elif 'int' in str(col_type):
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
            elif 'float' in str(col_type):
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        return df_optimized
    
    @staticmethod
    def chunk_processor(file_path: str, chunk_size: int = 10000, 
                       process_func: callable = None) -> pd.DataFrame:
        chunks = []
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            if process_func:
                chunk = process_func(chunk)
            chunks.append(chunk)
        
        return pd.concat(chunks, ignore_index=True)
    
    @staticmethod
    def parallel_apply(df: pd.DataFrame, func: callable, 
                      n_cores: int = -1) -> pd.DataFrame:
        from multiprocessing import Pool, cpu_count
        
        if n_cores == -1:
            n_cores = cpu_count()
        
        df_split = np.array_split(df, n_cores)
        
        with Pool(n_cores) as pool:
            results = pool.map(func, df_split)
        
        return pd.concat(results, ignore_index=True)

class ConfigManager:
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        import json
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
        level = getattr(logging, log_level.upper())
        
        handlers = [logging.StreamHandler()]
        if log_file:
            handlers.append(logging.FileHandler(log_file))
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
