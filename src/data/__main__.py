from .data_loader import DataLoader
from .data_cleaner import DataCleaner


def main():
    '''Main function to run the data pipeline'''
    
    print('Starting data pipeline...')
    
    
    #Load Data
    loader=DataLoader()
    raw_data = loader.load_data()
    
    #clean Data
    cleaner=DataCleaner()
    clean_data=cleaner.clean_data(raw_data)
    
    print('Data pipeline completed successfully')
    
    return clean_data




if __name__=='__main__':
    main()
    
    