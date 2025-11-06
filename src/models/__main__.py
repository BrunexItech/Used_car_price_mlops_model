from .model_trainer import ModelTrainer
from src.features.__main__ import main as feature_pipeline

def main():
    '''Main function to run model training pipeline'''
    print('Starting model training pipeline...')
    
    #Get features from feature pipeline
    X_train_scaled,X_test_scaled,y_train,y_test,engineer = feature_pipeline()
    
    
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
    from src.features.__main__ import save_encoders
    save_encoders(engineer)
    
    
    print('Model training pipeline completed successfully')
    return trainer, engineer


if __name__=='__main__':
    main()
