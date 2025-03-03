import os
import pandas as pd
import kagglehub
import shutil
import gdown
from src.configuration.config import ConfigurationManager
from src.logger import logger
from datetime import datetime

class RawDataStorage:
    def __init__(self,kaggle_df,gdrive_df):
        self.config=ConfigurationManager().get_data_ingestion_config()
        self.kaggle_df=kaggle_df
        self.gdrive_df=gdrive_df
    def run_data_storage(self):
        kaggle_df=self.kaggle_df
        gdrive_df=self.gdrive_df
        if(os.path.exists(self.config.staging_kaggle) and os.path.isdir(self.config.staging_kaggle)):
            shutil.rmtree(self.config.staging_kaggle)
        if(os.path.exists(self.config.staging_gdrive) and os.path.isdir(self.config.staging_gdrive)):
            shutil.rmtree(self.config.staging_gdrive)
        os.makedirs(self.config.staging_kaggle)
        os.makedirs(self.config.staging_gdrive)
        
        return kaggle_df.to_csv(self.config.staging_kaggle+"raw_kaggle_churn_data.csv",index=False),gdrive_df.to_csv(self.config.staging_gdrive+"raw_gdrive_churn_data.csv",index=False)