import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional
from datetime import datetime, timedelta

class TimeSeriesProcessor:
    
    @staticmethod
    def set_datetime_index(df: pd.DataFrame, date_column: str, 
                          freq: Optional[str] = None) -> pd.DataFrame:
        df_ts = df.copy()
        df_ts[date_column] = pd.to_datetime(df_ts[date_column])
        df_ts = df_ts.set_index(date_column)
        
        if freq:
            df_ts = df_ts.asfreq(freq)
        
        return df_ts
    
    @staticmethod
    def resample_data(df: pd.DataFrame, rule: str, agg_func: str = 'mean') -> pd.DataFrame:
        agg_funcs = {
            'mean': lambda x: x.mean(),
            'sum': lambda x: x.sum(),
            'min': lambda x: x.min(),
            'max': lambda x: x.max(),
            'count': lambda x: x.count()
        }
        
        return df.resample(rule).agg(agg_funcs[agg_func])
    
    @staticmethod
    def fill_missing_timestamps(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        df_filled = df.copy()
        
        if method == 'interpolate':
            df_filled = df_filled.interpolate()
        elif method == 'forward_fill':
            df_filled = df_filled.fillna(method='ffill')
        elif method == 'backward_fill':
            df_filled = df_filled.fillna(method='bfill')
        
        return df_filled
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, columns: List[str], 
                           lags: List[int]) -> pd.DataFrame:
        df_lagged = df.copy()
        
        for col in columns:
            for lag in lags:
                df_lagged[f'{col}_lag_{lag}'] = df_lagged[col].shift(lag)
        
        return df_lagged
    
    @staticmethod
    def rolling_statistics(df: pd.DataFrame, columns: List[str], 
                          window: int, stats: List[str] = ['mean']) -> pd.DataFrame:
        df_rolling = df.copy()
        
        for col in columns:
            for stat in stats:
                if stat == 'mean':
                    df_rolling[f'{col}_rolling_{window}_mean'] = df_rolling[col].rolling(window).mean()
                elif stat == 'std':
                    df_rolling[f'{col}_rolling_{window}_std'] = df_rolling[col].rolling(window).std()
                elif stat == 'min':
                    df_rolling[f'{col}_rolling_{window}_min'] = df_rolling[col].rolling(window).min()
                elif stat == 'max':
                    df_rolling[f'{col}_rolling_{window}_max'] = df_rolling[col].rolling(window).max()
        
        return df_rolling

class AggregationEngine:
    
    @staticmethod
    def group_and_aggregate(df: pd.DataFrame, group_by: List[str], 
                           agg_dict: Dict[str, List[str]]) -> pd.DataFrame:
        return df.groupby(group_by).agg(agg_dict).reset_index()
    
    @staticmethod
    def pivot_table_advanced(df: pd.DataFrame, index: List[str], 
                           columns: str, values: str, 
                           aggfunc: str = 'mean') -> pd.DataFrame:
        return df.pivot_table(index=index, columns=columns, 
                            values=values, aggfunc=aggfunc, fill_value=0)
    
    @staticmethod
    def cross_tabulation(df: pd.DataFrame, index: str, 
                        columns: str, normalize: Optional[str] = None) -> pd.DataFrame:
        return pd.crosstab(df[index], df[columns], normalize=normalize)
    
    @staticmethod
    def window_functions(df: pd.DataFrame, partition_by: List[str], 
                        order_by: str, value_col: str) -> pd.DataFrame:
        df_window = df.copy()
        
        df_window['row_number'] = df_window.groupby(partition_by)[order_by].rank(method='first')
        df_window['cumsum'] = df_window.groupby(partition_by)[value_col].cumsum()
        df_window['pct_rank'] = df_window.groupby(partition_by)[value_col].rank(pct=True)
        
        return df_window
