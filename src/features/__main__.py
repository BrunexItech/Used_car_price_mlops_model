from .feature_engineer import FeatureEngineer
from src.data.__main__ import main as data_pipeline


def main():
    '''Main function to run the feature engineerin pipeline'''
    print('Starting feature engineering pipeline...')
    
    #Get clean data from data pipeline
    clean_data=data_pipeline()
    print(f"clean_data type: {type(clean_data)}") 
    
    #Engineer features
    engineer = FeatureEngineer()
    X_train_scaled,X_test_scaled,y_train,y_test=engineer.prepare_features(clean_data)
    
    print('Feature engineering completed successfully')
    return X_train_scaled,X_test_scaled,y_train,y_test, engineer



if __name__=='__main__':
    main()
    