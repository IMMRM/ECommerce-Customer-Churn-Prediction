import shutil
from src.configuration.config import ConfigurationManager
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
from src.logger import logger



class DataPreparation:
    def __init__(self):
        self.config=ConfigurationManager().get_data_preparation_config()
    def run_data_preparations(self):
        
        kaggle_df=pd.read_csv(self.config.kaggle+'/raw_kaggle_churn_data.csv')
        gdrive_df=pd.read_csv(self.config.gdrive+'/raw_gdrive_churn_data.csv')
        
        combined_raw_df=pd.concat([kaggle_df,gdrive_df],axis=0)
        #filling missing numeric values
        combined_raw_df.fillna(combined_raw_df.median(numeric_only=True),inplace=True)
        #filling missing categorical values
        combined_raw_df.fillna(combined_raw_df.mode().iloc[0],inplace=True)
        #Convert 'data_last_updated_dt' to datetime format
        combined_raw_df['data_last_updated_dt']=pd.to_datetime(combined_raw_df['data_last_updated_dt'],errors='coerce')
        categorical_columns = ["PreferredLoginDevice", "PreferredPaymentMode", "Gender", "PreferedOrderCat", "MaritalStatus", "source"] # These are the categorical columns in our data
        encoder = LabelEncoder()
        for col in categorical_columns:
            combined_raw_df[col] = encoder.fit_transform(combined_raw_df[col])
        
        scaler=joblib.load(self.config.models_path+'/scaler.joblib','r')
        numerical_columns = ["Tenure", "WarehouseToHome", "HourSpendOnApp", "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount", "DaySinceLastOrder", "CashbackAmount"]
        combined_raw_df[numerical_columns] = scaler.fit_transform(combined_raw_df[numerical_columns])
        combined_raw_df['data_last_updated_year'] = combined_raw_df['data_last_updated_dt'].dt.year
        combined_raw_df.drop(columns=['data_last_updated_dt'], inplace=True)  # Remove original date column
        # Dropping one column out of each highly correlated column pairs we saw above:
        high_corr_cols=[
            'source',
            'CouponUsed'
        ]
        df_wo_multicollinearity=combined_raw_df.drop(high_corr_cols,axis=1)
        # Keep Only Features Correlated with Churn (> 0.15)
        corr_with_churn = df_wo_multicollinearity.corr()["Churn"].abs()
        selected_features = corr_with_churn[corr_with_churn >= 0.10].index.tolist()
        selected_features.append("CustomerID")
        df_wo_multicollinearity = df_wo_multicollinearity[selected_features]
        df_wo_multicollinearity.to_csv(self.config.processed+"/churn_data_prepared.csv",index=False)
        logger.info("Data Preparation Completed and Saved as 'churn_data_prepared.csv'")