import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

class DataPreprocessor:

    def __init__(self,processed_path = 'data/processed'):
        self.processed_path = processed_path
        self.scaler = StandardScaler()
        os.makedirs(processed_path,exist_ok=True)

    def check_missing_values(self,df):

        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("Missing values found:")
            print(missing[missing > 0])
        else:
            print("No Missing values found")
        
        return df
    
    def split_features_target(self,df):

        X = df.drop('target', axis=1)
        y = df['target']
        print(f"Fetures shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        return X, y
    
    def train_val_test_split(self, X, y, val_size=0.15, test_size=0.15, random_state=42):

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size,random_state=random_state, stratify=y
        )

        val_friction = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_friction ,random_state=random_state, stratify=y_temp
        )

        # train:70, val:15, test15
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):

        X_train_scaled = self.scaler.fit_transform(X_train) # learn transform
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled

