import mlflow
import mlflow.sklearn
from datetime import datetime


class ExperimentTracker:
    def __init__(self,experiment_name='Car_Price_Prediction'):
        self.experiment_name=experiment_name
        mlflow.set_experiment(experiment_name)
        
        
    def start_run(self,run_name=None):
        '''Start a new MLflow run'''
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.run=mlflow.start_run(run_name=run_name)
        return self.run
        
    def log_params(self,params):
        '''Log parameters to mlflow'''
        mlflow.log_params(params)
        
    def log_metrics(self,metrics):
        '''Log metrics to mlflow'''
        mlflow.log_metrics(metrics)
        
    def log_model(self,model,model_name):
        '''Log model to mlflow'''
        mlflow.sklearn.log_model(model,model_name)
        
    def end_run(self):
        '''End the current mlflow run'''
        mlflow.end_run()