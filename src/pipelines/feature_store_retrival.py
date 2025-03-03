from feast import FeatureStore
from src.configuration.config import ConfigurationManager
from src.logger import logger
from datetime import datetime
import pandas as pd




class FeatureRetrival:
    def __init__(self):
        self.config=ConfigurationManager().get_feature_config()
    def retrieve_features(self):
        store = FeatureStore(repo_path=self.config.repo_path)  # Assumes feature_store.yaml is in the current directory
        df=pd.read_parquet("data/processed/transformed_churn_data.parquet")
        logger.info(">>> Started materialization >>>")
        date=datetime.now()
        store.materialize(start_date=date,end_date=date)
        logger.info(">>> Materialization completed >>>")
        # Retrieve online features for a specific customer
        online_features = store.get_online_features(
            features=["customer_features:TotalSpend", "customer_features:ActivityFrequency"],
            entity_rows=[{"CustomerID": customer_id} for customer_id in df["CustomerID"]],  # Replace with a valid CustomerID from your data
        ).to_df()
        print(online_features)




