import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer

class DataCleaner:
    
    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None, 
                         keep: str = 'first') -> pd.DataFrame:
        return df.drop_duplicates(subset=subset, keep=keep)
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean', 
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        df_clean = df.copy()
        
        if strategy in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy)
            df_clean[columns] = imputer.fit_transform(df_clean[columns])
        elif strategy == 'knn':
            imputer = KNNImputer()
            df_clean[columns] = imputer.fit_transform(df_clean[columns])
        elif strategy == 'drop':
            df_clean = df_clean.dropna(subset=columns)
        
        return df_clean
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, columns: List[str], 
                       method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
        df_clean = df.copy()
        
        for col in columns:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - factor * IQR
                upper = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                df_clean = df_clean[z_scores < factor]
        
        return df_clean
    
    @staticmethod
    def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        df_clean.columns = (df_clean.columns
                           .str.lower()
                           .str.replace(' ', '_')
                           .str.replace('[^a-z0-9_]', '', regex=True))
        return df_clean

class DataTransformer:
    
    @staticmethod
    def scale_features(df: pd.DataFrame, columns: List[str], 
                      method: str = 'standard') -> Tuple[pd.DataFrame, Any]:
        df_scaled = df.copy()
        
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        scaler = scalers[method]
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        
        return df_scaled, scaler
    
    @staticmethod
    def create_datetime_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        df_features = df.copy()
        df_features[date_column] = pd.to_datetime(df_features[date_column])
        
        df_features[f'{date_column}_year'] = df_features[date_column].dt.year
        df_features[f'{date_column}_month'] = df_features[date_column].dt.month
        df_features[f'{date_column}_day'] = df_features[date_column].dt.day
        df_features[f'{date_column}_dayofweek'] = df_features[date_column].dt.dayofweek
        df_features[f'{date_column}_quarter'] = df_features[date_column].dt.quarter
        
        return df_features
    
    @staticmethod
    def encode_categorical(df: pd.DataFrame, columns: List[str], 
                          method: str = 'onehot') -> pd.DataFrame:
        df_encoded = df.copy()
        
        if method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=True)
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            for col in columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        return df_encoded
