import os
import pandas as pd
import pyodbc
from src.logger import logger
import sqlite3
from src.configuration.config import ConfigurationManager
from datetime import datetime

class DataTransformationStorage:
    def __init__(self):
        self.config=ConfigurationManager().get_data_storage_config()
        #self.data_config=ConfigurationManager().get_data_ingestion_config()
        print(self.config)
    def store_in_sql(self,df):
        # SQLite Database Setup
        try:
            conn = sqlite3.connect(self.config.db_path)
            logger.info("Connection to DB successfully established!")
        except Exception as e:
            raise e
        cursor = conn.cursor()

        # Create Table for Transformed Data
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {self.config.data_table_name} (
            CustomerID INTEGER PRIMARY KEY,
            Churn INTEGER,
            Tenure REAL,
            NumberOfDeviceRegistered INTEGER,
            PreferedOrderCat INTEGER,
            SatisfactionScore INTEGER,
            MaritalStatus INTEGER,
            Complain INTEGER,
            DaySinceLastOrder REAL,
            CashbackAmount REAL,
            TotalSpend REAL,
            ActivityFrequency REAL,
            IsLongTermCustomer INTEGER
        );
        ''')
        try:
            logger.info("DataLoad Started!")
            # Insert Transformed Data into Database
            df.to_sql(self.config.data_table_name, conn, if_exists="replace", index=False)
            logger.info("DataLoad Completed!")
        except Exception as e:
            raise e
        conn.close()
    
    def data_transformation(self):
        logger.info("Data Transformation Started!")
        df=pd.read_csv(self.config.processed+"churn_data_prepared.csv")
        # Feature Engineering
        # Aggregated Feature: Total spend per customer (Assuming CashbackAmount as a proxy for spend)
        df["TotalSpend"] = df["CashbackAmount"] * df["Tenure"]

        # Derived Feature: Activity Frequency
        df["ActivityFrequency"] = df["NumberOfDeviceRegistered"] / df["Tenure"].replace(0, 1)
        logger.info("Data Transformation Completed!")
        try:
            logger.info("Transformed Data Storage Started!")
            self.store_in_sql(df)
            # Add a synthetic event timestamp (assuming data is recent)
            df["event_timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            df["event_timestamp"] = pd.to_datetime(df["event_timestamp"]).dt.tz_localize("UTC")
            df.to_parquet(self.config.processed+"transformed_churn_data.parquet")
            logger.info("Transformed Data Storage Completed!")
        except Exception as e:
            raise e

        
        
        