import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from ..config import TARGET_COLUMN, MODEL_CONFIG


class FeatureEngineer:
    def __init__(self):
        self.label_encoders={}
        self.scaler=StandardScaler()
        self.feature_names=None
        
        
    def encode_categorical(self,clean_data):
        '''Encoding categorical variables using LabelEncoder'''
        encoded_data=clean_data.copy()
        categorical_cols = encoded_data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le=LabelEncoder()
            encoded_data[col]=le.fit_transform(encoded_data[col].astype(str))
            self.label_encoders[col]=le
            
        print(f'Encoded{len(categorical_cols)} categorical columns')
        return encoded_data
    
    
    
    #Splitting the data
    def split_features_target(self, encoded_data):
        '''Separate features and target variables'''
        X=encoded_data.drop(columns=[TARGET_COLUMN])
        y=encoded_data[TARGET_COLUMN]
        self.feature_names=X.columns.tolist()
        return X,y
    
    
    #Normalization
    def scaled_features(self,X_train,X_test):
        '''Scale features using Standardscaler'''
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled=self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
        
        
    def prepare_features(self,clean_data):
        '''Complete feature engineering pipeline on clean data'''
        #Encode Categorical variables
        encoded_data=self.encode_categorical(clean_data)
        
        #Separate features and target 
        X,y = self.split_features_target(encoded_data)
        
        #Split data into train/test
        X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                         test_size=MODEL_CONFIG['test_size'],
                                                         random_state=MODEL_CONFIG['random_state'])
        #Scale features
        X_train_scaled,X_test_scaled=self.scaled_features(X_train,X_test)
        
        print(f'final featurs:{len(self.feature_names)} columns')
        print(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
        
        return X_train_scaled,X_test_scaled,y_train,y_test
