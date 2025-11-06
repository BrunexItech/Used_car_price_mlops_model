from .data_loader import DataLoader
from .data_cleaner import DataCleaner
from .data_validator import DataValidator


def main():
    '''Main function to run the data pipeline'''
    
    print('Starting data pipeline...')
    
    
    #Load Data
    loader=DataLoader()
    raw_data = loader.load_data()
    
    #Validate Data
    print('Validating data quality ...')
    validator = DataValidator()
    is_valid=validator.validate_raw_data(raw_data)
    
    if not is_valid:
        raise ValueError('Data validation failed! Check data quality.')
    
    #clean Data
    cleaner=DataCleaner()
    clean_data=cleaner.clean_data(raw_data)
    
    print('Data pipeline completed successfully')
    
    return clean_data




if __name__=='__main__':
    main()
    
    