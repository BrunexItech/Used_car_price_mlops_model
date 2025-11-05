import pandas as pd
from ..config import RAW_DATA_PATH


class DataLoader:
    def __init__(self):
        self.file_path=RAW_DATA_PATH
        
        
    def load_data(self):
        df=pd.read_csv(self.file_path)
        print(f'Loaded data shape:{df.shape}')
        return df