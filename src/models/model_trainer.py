from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error
import joblib 
import numpy as np
from ..config import MODELS_DIR



class ModelTrainer:
    def __init__(self):
        self.models={}
        self.best_model_name=None
        self.best_model=None
        self.best_score=0
        self.final_model=None
        
    def train_models(self,X_train,y_train):
        '''Train multiple linear models and compare performance'''
        linear_models={
            'Linear Regression':LinearRegression(),
            'Ridge':Ridge(alpha=1.0),
            'Lasso':Lasso(alpha=1.0),
            'ElasticNet':ElasticNet(alpha=1.0, l1_ratio=0.5)
            
        }
        #Evaluating models using cross-val-score
        for name, model in linear_models.items():
            scores=cross_val_score(model,X_train,y_train,cv=5, scoring='r2')
            mean_score=scores.mean()
            self.models[name]={
                'model':model,
                'cv_score':mean_score,
                'std':scores.std()
            }
            print(f'{name}:R²={mean_score:.4f} ')
            
            
            #Track the best model
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_model_name=name
                self.best_model = model
    
        print(f'Best model:{self.best_model_name} with R²: {self.best_score:.4f} ')
        return self.models
    
    def fine_tune_best_model(self,X_train,y_train):
        '''Fine tune the best performing model'''
        if self.best_model_name == 'Ridge':
            param_grid={'alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        elif self.best_model_name == 'Lasso':
            param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        elif self.best_model_name == 'ElasticNet':
            param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
        else: #Linear regression has no hyperparameters
            self.final_model=self.best_model.fit(X_train,y_train)
            return self.final_model
        
        
        #Using Gridsearch CV
        grid_search=GridSearchCV(self.best_model,param_grid,cv=5,scoring='r2')
        grid_search.fit(X_train,y_train)
        
        
        self.final_model=grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best R² after tuning: {grid_search.best_score_:.4f}")
        
        return self.final_model
    
    
    
    #Evaluating the final model
    def evaluate_final_model(self,X_test,y_test):
        '''Evaluate the final tuned model on test set'''
        y_pred=self.final_model.predict(X_test)
        r2=r2_score(y_test,y_pred)
        rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    
    
        print(f'Final Model Test  R²: {r2:.4f}')
        print(f"Final Model Test RMSE: {rmse:.4f}")
        
        
        return r2,rmse
    
    
    def save_model(self,filename='car_price_model.joblib'):
        '''Save the final trained model'''
        model_path=MODELS_DIR/filename
        joblib.dump(self.final_model,model_path)
        print(f'Final model saved to {model_path}')
    