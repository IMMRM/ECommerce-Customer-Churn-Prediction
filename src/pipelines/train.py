import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import joblib
from src.utils import common
from src.configuration.config import ConfigurationManager
import os
from sklearn.model_selection import (GridSearchCV,train_test_split)
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,precision_score,recall_score,f1_score)
from src.logger import logger
from urllib.parse import urlparse
from pathlib import Path


class Train:
    def __init__(self):
        self.mlflow_creds=ConfigurationManager().get_mlflow_secret_config()
        self.mlflow_params=ConfigurationManager().get_train_params()
        self.mlflow_hyperparams_grid=ConfigurationManager().get_hyperparameter_grid()
        os.environ["MLFLOW_TRACKING_URI"]=self.mlflow_creds.tracking_ui
        os.environ["MLFLOW_TRACKING_USERNAME"]=self.mlflow_creds.username
        os.environ["MLFLOW_TRACKING_PASSWORD"]=self.mlflow_creds.password
    def evaluate_metrics(self,y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        return accuracy, precision, recall, f1
    def hyperparameter_tuning(self,X_train,y_train,params,model):
        #rf=RandomForestClassifier()
        grid_search=GridSearchCV(estimator=model,param_grid=params,n_jobs=-1,cv=3)
        grid_search.fit(X_train,y_train)
        return grid_search
    def train(self,data_path,model_path,random_state,n_estimators,max_depth):
        df=pd.read_csv(self.mlflow_params.data_path)
        X=df.drop(columns=["Churn"])
        Y=df["Churn"]
        #set the tracking uri for mlflow
        mlflow.set_tracking_uri(self.mlflow_creds.tracking_ui)
        mlflow.set_experiment("Experiment V1")
        # Define models to train
        models = {
            "RandomForest": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(),
            "GradientBoosting": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier()
        }
        #start with MLFlow run
        with mlflow.start_run():
            X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20)
            signature=infer_signature(X_train,Y_train)
            for model_name,model in models.items():
                with mlflow.start_run(nested=True,run_name=model_name):
                    logger.info(f"Training {model_name}")
                    #Hyperparameter tuning (if applicable)
                    if(model_name in self.mlflow_hyperparams_grid):
                        print(model_name)
                        print(self.mlflow_hyperparams_grid[model_name])
                        gridsearch=self.hyperparameter_tuning(X_train=X_train,y_train=Y_train,params=self.mlflow_hyperparams_grid[model_name],model=model)
                        best_model=gridsearch.best_estimator_
                        
                        #Log best hyperparameters
                        for param,value in gridsearch.best_params_.items():
                            mlflow.log_param(f"best_{param}",value)
                    else:
                        best_model=model.fit(X_train,Y_train)
                        #log default parameters
                        for param,value in model.get_params().items():
                            mlflow.log_param(param,value)
                    
                    y_pred=best_model.predict(X_test)
                    accuracy,precision,recall,f1=self.evaluate_metrics(Y_test,y_pred)
                    #Log metrics
                    mlflow.log_metric("accuracy",accuracy)
                    mlflow.log_metric("precision",precision)
                    mlflow.log_metric("recall",recall)
                    mlflow.log_metric("f1_score",f1)
                    
                     #log the confusion matrix and classification report
                    cm=confusion_matrix(Y_test,y_pred)
                    cr=classification_report(Y_test,y_pred)
                    
                    mlflow.log_text(str(cm),f"{model_name}_confusion_matrix.txt")
                    mlflow.log_text(cr,f"{model_name}_classification_report.txt")
                    
                    tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
            
                    if(tracking_url_type_store!="file"):
                        mlflow.sklearn.log_model(
                            sk_model=best_model,
                            artifact_path=f"{model_name}_model_1",
                            registered_model_name=f"{model_name}_best model"
                        )
                    else:
                        mlflow.sklearn.log_model(
                            sk_model=best_model,
                            artifact_path=f"{model_name}_model_1",
                            signature=signature
                        )
                        
                    joblib.dump(best_model,Path(self.mlflow_params.model_path)/f"{model_name}_model.joblib")
                    logger.info(f"{model_name} model saved to model path")
                    
            # param_grid={
            #     'n_estimators': [100,200],
            #     'max_depth': [5,10,None],
            #     'min_samples_split':[2,5],
            #     'min_samples_leaf':[1,2]
            # }
            # #Perform hyperparameter tuning
            # gridsearch=self.hyperparameter_tuning(X_train=X_train,y_train=Y_train,params=param_grid)
            
            # #get the best model
            # best_model=gridsearch.best_estimator_
            
            # #predict and evaluate the model
            # y_pred=best_model.predict(X_test)
            
            # #Calculating the accuracy score
            # accuracy=accuracy_score(Y_test,y_pred)
            
            # #logging the accuracy in our local log files
            # logger.info(f"Accuracy score: {accuracy}")
            
            # #logging additional metrics on mlflow
            # mlflow.log_metric("accuracy",accuracy)
            # #logging parameters on mlflow
            # mlflow.log_param("best_n_estimators",gridsearch.best_params_['n_estimators'])
            # mlflow.log_param("best max_depth",gridsearch.best_params_['max_depth'])
            # mlflow.log_param("best min_samples_split",gridsearch.best_params_['min_samples_split'])
            # mlflow.log_param("best min_samples_leaf",gridsearch.best_params_['min_samples_leaf'])
            
            # #log the confusion matrix and classification report
            # cm=confusion_matrix(Y_test,y_pred)
            # cr=classification_report(Y_test,y_pred)
            
            # mlflow.log_text(str(cm),"confusion matrix.txt")
            # mlflow.log_text(cr,"classification report.txt")
            
            # tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
            
            # if(tracking_url_type_store!="file"):
            #     mlflow.sklearn.log_model(
            #         sk_model=best_model,
            #         artifact_path="model_1",
            #         registered_model_name="best model"
            #     )
            # else:
            #     mlflow.sklearn.log_model(
            #         sk_model=best_model,
            #         artifact_path="model_1",
            #         signature=signature
            #     )
                
            # joblib.dump(best_model,Path(self.mlflow_params.model_path))
            
            # logger.info("Model saved to model path!")
    def run_train(self):
        self.train(self.mlflow_params.data_path,self.mlflow_params.model_path,self.mlflow_params.random_state,self.mlflow_params.n_estimators,self.mlflow_params.max_depth)
            
            
            
            
            
            
            
        
        
        