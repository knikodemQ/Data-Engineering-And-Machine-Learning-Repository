import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

class DataLoader:
    
    @staticmethod
    def load_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        return pd.read_csv(file_path, **kwargs)
    
    @staticmethod
    def load_json(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        return pd.read_json(file_path, **kwargs)
    
    @staticmethod
    def load_pickle(file_path: Union[str, Path]) -> pd.DataFrame:
        return pd.read_pickle(file_path)
    
    @staticmethod
    def load_excel(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        return pd.read_excel(file_path, **kwargs)
    
    @staticmethod
    def auto_load(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        path = Path(file_path)
        ext = path.suffix.lower()
        
        loaders = {
            '.csv': DataLoader.load_csv,
            '.json': DataLoader.load_json,
            '.pkl': DataLoader.load_pickle,
            '.pickle': DataLoader.load_pickle,
            '.xlsx': DataLoader.load_excel,
            '.xls': DataLoader.load_excel
        }
        
        if ext not in loaders:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return loaders[ext](file_path, **kwargs)

class DataSaver:
    
    @staticmethod
    def save_csv(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
        df.to_csv(file_path, index=False, **kwargs)
    
    @staticmethod
    def save_json(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
        df.to_json(file_path, **kwargs)
    
    @staticmethod
    def save_pickle(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
        df.to_pickle(file_path)
    
    @staticmethod
    def save_excel(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
        df.to_excel(file_path, index=False, **kwargs)
    
    @staticmethod
    def save_multiple_formats(df: pd.DataFrame, base_path: Union[str, Path], 
                            formats: List[str] = ['csv', 'json', 'pickle']) -> None:
        base = Path(base_path)
        
        for fmt in formats:
            if fmt == 'csv':
                DataSaver.save_csv(df, f"{base}.csv")
            elif fmt == 'json':
                DataSaver.save_json(df, f"{base}.json")
            elif fmt == 'pickle':
                DataSaver.save_pickle(df, f"{base}.pkl")
            elif fmt == 'excel':
                DataSaver.save_excel(df, f"{base}.xlsx")
