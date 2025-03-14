from src.constants import CONFIG_PATH,PARAMS_PATH,SCHEMA_PATH,SECRETS_PATH
from src.utils.common import read_yaml
from src.entity.data_ingestion import DataIngestionConfig
from src.entity.data_storage import DataStorageConnectionConfig
from src.entity.data_validation import DataValidationConfig
from src.entity.data_preparations import DataPreparationConfig
from src.entity.feature_store import FeatureRetrivalConfig
from src.entity.mlflow import (MLFlowCreds,Parameters)




class ConfigurationManager:
    def __init__(self,config_path=CONFIG_PATH,params_path=PARAMS_PATH,schema_path=SCHEMA_PATH,secret_path=SECRETS_PATH):
        self.config=read_yaml(config_path),
        self.params=read_yaml(params_path),
        self.schema=read_yaml(schema_path),
        self.secrets=read_yaml(secret_path)
    def get_data_ingestion_config(self)->DataIngestionConfig:
        config=self.config[0].data
        data_ingestion=DataIngestionConfig(
            kaggle_dataset=config.kaggle_source_path,
            gdrive_dataset=config.gdrive_source_path,
            local_dataset=config.raw,
            staging_dataset=config.interim,
            staging_kaggle=config.interim_kaggle,
            staging_gdrive=config.interim_gdrive
        )
        return data_ingestion
    def get_data_storage_config(self)->DataStorageConnectionConfig:
        config=self.config[0].sql
        data_connection=DataStorageConnectionConfig(
            db_path=config.db_path,
            processed=config.processed,
            data_table_name=config.data_table_name
        )
        return data_connection
    def get_data_validation_config(self)->DataValidationConfig:
        config=self.config[0].data_validation,
        schema=self.schema.COLUMNS
        data_validation=DataValidationConfig(
            data_source_path_kaggle=config[0].data_source_kaggle,
            data_source_path_gdrive=config[0].data_source_gdrive,
            Status_report=config[0].STATUS_REPORT_FILE,
            all_schema=schema
        )
        return data_validation
    def get_data_preparation_config(self)->DataPreparationConfig:
        config=self.config[0].data
        data_preparation=DataPreparationConfig(
            kaggle=config.interim_kaggle,
            gdrive=config.interim_gdrive,
            models_path=config.models,
            processed=config.processed
        )
        return data_preparation
    def get_feature_config(self)->FeatureRetrivalConfig:
        config=self.config[0].feature_store_repo
        feature_config=FeatureRetrivalConfig(
            repo_path=config.repo_path
        )
        return feature_config
    def get_mlflow_secret_config(self)->MLFlowCreds:
        secrets=self.secrets.ml_flow_creds
        mlflow_creds=MLFlowCreds(
            tracking_ui=secrets.tracking_uri,
            username=secrets.username,
            password=secrets.password
        )
        return mlflow_creds
    def get_train_params(self)->Parameters:
        parameters=self.params[0].train
        params=Parameters(
            data_path=parameters.data,
            model_path=parameters.model,
            random_state=parameters.random_state,
            n_estimators=parameters.n_estimators,
            max_depth=parameters.max_depth
        )
        return params
    def get_hyperparameter_grid(self):
        grid=self.params[0].hyperparameter_grids
        return grid