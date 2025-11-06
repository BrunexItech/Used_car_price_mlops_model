from .feature_engineer import FeatureEngineer
from src.data.__main__ import main as data_pipeline
from ..config import MODELS_DIR
import joblib

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


#add function to save encoders
def save_encoders(engineer, filename='label_encoders.joblib'):
    '''Save label encoders to file'''
    encoders_path = MODELS_DIR/filename
    joblib.dump(engineer.label_encoders, encoders_path)
    print(f'Label encoders saved to {encoders_path}')


if __name__=='__main__':
    main()
    