import great_expectations as ge
import pandas as pd
import logging
from ..config import RAW_DATA_PATH


logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)


class DataValidator:
    def __init__(self):
        self.suite_name = 'car_price_data_suite'
        
    def validate_raw_data(self,df):
        '''Validate raw data against expectations'''
        
        try:
            logger.info('Starting data validation...')
            
            
            #convert to Great Exapectations dataset
            ge_df = ge.from_pandas(df)
            
            
            #Define data quality expectations
            results = ge_df.expect_column_to_exist('Price')
            results = ge_df.expect_column_values_to_be_between('Price',0,500000)
            results = ge_df.expect_column_to_exist("Mileage(miles)")
            results = ge_df.expect_column_values_to_be_between("Mileage(miles)", 0, 500000)
            results = ge_df.expect_column_to_exist("Registration_Year") 
            results = ge_df.expect_column_values_to_be_between("Registration_Year", 1990, 2024)
            results = ge_df.expect_column_values_to_not_be_null("Engine")
            results = ge_df.expect_column_values_to_be_in_set("Fuel type", ["Diesel", "Petrol", "Petrol Hybrid", "Petrol Plug-in Hybrid"])
            results = ge_df.expect_column_values_to_be_in_set("Gearbox", ["Automatic", "Manual"])
            
            
            #Check if validation passed
            if results.success:
                logger.info('Data validation passed')
                return True
            else:
                logger.error('Data validation failed')
                for result in results.results:
                    if not result.success:
                        logger.error(f'Failed expectation:{result.expectation_config.expectaion_type}')
                        return False
                    
        except Exception as e:
            logger.error(f'Data validation error : {e}')
            return False