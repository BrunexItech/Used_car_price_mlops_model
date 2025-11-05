import pandas as pd
import numpy as np
from ..config import DROP_COLUMNS


class DataCleaner:
    def __init__(self):
        self.drop_columns=DROP_COLUMNS
        
        
    #This is a function to remove any outliers present 
    def _remove_outliers(self, df, column_name):
        """Remove outliers from a specific column using IQR method"""
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_no_outliers = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
        outliers_removed = df.shape[0] - df_no_outliers.shape[0]
        
        print(f"Removed {outliers_removed} outliers from {column_name}")
        return df_no_outliers
        
    def clean_data(self,df):
        '''Cleaning the raw data'''
        df_clean=df.copy()
        
        #Drop the specified columns
        df_clean=df_clean.drop(columns=self.drop_columns, errors='ignore')
        
        #Handle missing values
        if 'Previous Owners' in df_clean.columns:
            previous_owners_median=df_clean['Previous Owners'].median()
            df_clean['Previous Owners']=df_clean['Previous Owners'].fillna(previous_owners_median)
            print(f'Filled missing Previous owners with median')
        
        #Handle Engine columns -remove L and convert to float
        if 'Engine' in df_clean.columns:
            df_clean['Engine']=df_clean['Engine'].str.replace('L','').astype(float)
            print('Cleaned Engine column')
            
            
        #Drop rows with null values in specific columns
        rows_to_drop=['Engine', 'Doors', 'Seats']
        df_clean = df_clean.dropna(subset=rows_to_drop)
        print('Null rows dropped')
        
        
        #Remove outliers using IQR method
        df_clean=self._remove_outliers(df_clean, 'Mileage(miles)')
        df_clean=self._remove_outliers(df_clean, 'Price')
        print('Removed all the outliers')
        
        
        
        
        return df_clean