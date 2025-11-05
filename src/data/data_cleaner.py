import pandas as pd
import numpy as np
from ..config import DROP_COLUMNS


class DataCleaner:
    def __init__(self):
        self.drop_columns=DROP_COLUMNS
        
        
    def clean_data(self,df):
        '''Cleaning the raw data'''
        df_clean=df.copy()
        
        #Drop the specified columns
        df_clean=df_clean.drop(columns=self.drop_columns, errors='ignore')
        
        
        return df_clean