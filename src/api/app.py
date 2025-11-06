from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from ..config import MODELS_DIR


#Load the trained model
model_path=MODELS_DIR/'car_price_model.joblib'
model=joblib.load(model_path)

#The label encoders
encoders_path=MODELS_DIR/'label_encoders.joblib'
label_encoders = joblib.load(encoders_path)

app=FastAPI(title='Car Price Prediction API', version='1.0.0')

class CarFeatures(BaseModel):
    '''Input features for car price predicton -exact columns'''
    title:str
    mileage:float
    registration_year:int
    previous_owners:int
    fuel_type:str
    body_type:str
    engine:float
    gearbox:str
    doors:int
    seats:int
    
    
@app.get('/')
def read_root():
    return{'message':'Car Prediction API'}

@app.get('/health')
def health_check():
    return {'status':'healthy'}

@app.post('/predict')
def predict_price(features:CarFeatures):
    '''Predict car price based on raw input features'''
    try:
        #Encode categorical features using saved label encoders
        title_encoded=label_encoders['title'].transform([features.title])[0]
        fuel_type_encoded = label_encoders['Fuel type'].transform([features.fuel_type])[0]
        body_type_encoded = label_encoders['Body type'].transform([features.body_type])[0]
        gearbox_encoded = label_encoders['Gearbox'].transform([features.gearbox])[0]
        
        
        #convert input to numpy arrray for prediction
        input_features=np.array([[
            title_encoded,
            features.mileage,
            features.registration_year,
            features.previous_owners,
            fuel_type_encoded,
            body_type_encoded,
            features.engine,
            gearbox_encoded,
            features.doors,
            features.seats
            
        ]])
        
        
        prediction=model.predict(input_features)
        
        
        return{
            'predicted_price':float(prediction[0]),
            'status':'success'
        }
        
    except Exception as e:
        return{'error':str(e),'status':'error'}

    