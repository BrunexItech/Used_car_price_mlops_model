from .model_trainer import ModelTrainer
from src.features.__main__ import main as feature_pipeline
from monitoring.drift_detector import DriftDetector
from src.data.__main__ import main as data_pipeline
import joblib


def main():
    '''Main function to run model training pipeline'''
    print('Starting model training pipeline...')
    
    #Get features from feature pipeline
    X_train_scaled,X_test_scaled,y_train,y_test,engineer = feature_pipeline()
    
    
    #DRIFT DETECTOR
    #Save reference data for drift detection
    drift_detector=DriftDetector()
    
    #Get the encoded data before scaling for drift detection
    clean_data = data_pipeline()
    encoded_data = engineer.encode_categorical(clean_data)
    drift_detector.set_reference_data(encoded_data)
    
    #save drift detector 
    joblib.dump(drift_detector, 'models/drift_detector.joblib')
    print('Drift detector saved with reference data')
    
    
    #Training and evaluating models
    trainer=ModelTrainer()
    
    
    #1. Train multiple models and select the best
    print(f'\n Training multiple models')
    models=trainer.train_models(X_train_scaled,y_train)
    
    
    #Fine tune the best model
    print(f'\n  Fine tuning Best Model')
    final_model=trainer.fine_tune_best_model(X_train_scaled,y_train)
    
    #Evaluate on test set
    print(f'Final model evaluation')
    r2,rmse=trainer.evaluate_final_model(X_test_scaled,y_test)
    
    
    #Save the model
    trainer.save_model()
    from src.features.__main__ import save_encoders_and_scaler
    save_encoders_and_scaler(engineer)
    
    
    print('Model training pipeline completed successfully')
    return trainer, engineer


if __name__=='__main__':
    main()
