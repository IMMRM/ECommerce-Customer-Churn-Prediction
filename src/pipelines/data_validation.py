import shutil
from src.configuration.config import ConfigurationManager
import os
import great_expectations as gx 
from great_expectations import expectations as gxe
import pandas as pd
from great_expectations.checkpoint import UpdateDataDocsAction

class DataValidation:
    def __init__(self):
        self.config=ConfigurationManager().get_data_validation_config()
    def run_validation(self):
        #Create a context
        if(os.path.exists("gx/") and os.path.isdir("gx/")):
            shutil.rmtree("gx/")
        context=gx.get_context(mode='file')
        #read data into dataframe
        df_kaggle = pd.read_csv(self.config.data_source_path_kaggle+"raw_kaggle_churn_data.csv")
        df_gdrive = pd.read_csv(self.config.data_source_path_gdrive+"raw_gdrive_churn_data.csv")
        combined_df=pd.concat([df_kaggle,df_gdrive],axis=0)
        #giving a data source name to identify all data sources uniquely
        datasource = context.data_sources.add_pandas(name="churn_data_source")

        # Create a data asset from the DataFrame
        data_asset = datasource.add_dataframe_asset(name="churn_data_asset")

        # Build a batch request using whole dataframe
        batch_definition = data_asset.add_batch_definition_whole_dataframe("bulk batch")
        batch = batch_definition.get_batch(batch_parameters={"dataframe": combined_df})
        
        #Organize expectation suite
        col_existence_suite_name="Column existence validation suite"
        col_suite=gx.ExpectationSuite(name=col_existence_suite_name)
        
        is_datatype_same_suite="Data Type Check Suite"
        type_suite=gx.ExpectationSuite(name=is_datatype_same_suite)
        suite_1=context.suites.add(col_suite)
        suite_2=context.suites.add(type_suite)
        # expectation=gx.expectations.ExpectColumnValuesToBeBetween(
        #     column="WarehouseToHome", min_value=5, max_value=126
        # )
        schema=self.config.all_schema
        for col in schema.keys():
                expectation1=gx.expectations.ExpectColumnToExist(column=col)
                suite_1.add_expectation(expectation1)
        expectation1.save()
        for col,dtype in schema.items():
            try:
                expectation2=gx.expectations.ExpectColumnValuesToBeOfType(column=col,type_=dtype)
                suite_2.add_expectation(expectation2)
            except Exception as e:
                raise e
                
            
        #suite.add_expectation(expectation)
        expectation2.save()
        # validation definition name
        validation_definition_name_col="dataset_cols"
        validation_definition_cols=gx.ValidationDefinition(
            data=batch_definition, suite=suite_1,name=validation_definition_name_col
        )
        validation_definition_name_type="dataset_types"
        validation_definition_types=gx.ValidationDefinition(
            data=batch_definition, suite=suite_2,name=validation_definition_name_type
        )
        context.validation_definitions.add(validation_definition_cols)
        context.validation_definitions.add(validation_definition_types)
        action_list=[
        UpdateDataDocsAction(
        name="update_all_data_docs"
         )
        ]
        #create checkpoint
        validation_defintions=[context.validation_definitions.get("dataset_types"),
        context.validation_definitions.get("dataset_cols")]
        checkpoint_name="first check" 
        checkpoint=gx.Checkpoint(
            name=checkpoint_name,
            validation_definitions=validation_defintions,
            actions=action_list,
            result_format={"result_format":"COMPLETE"}
        )

        #save the checkpoint
        context.checkpoints.add(checkpoint)

        validation_result=checkpoint.run(
            batch_parameters={"dataframe":combined_df}
        )


        #Configure Data Docs
        base_directory = "uncommitted/data_docs/local_site/"  # this is the default path (relative to the root folder of the Data Context) but can be changed as required
        site_config = {
            "class_name": "SiteBuilder",
            "site_index_builder": {"class_name": "DefaultSiteIndexBuilder"},
            "store_backend": {
                "class_name": "TupleFilesystemStoreBackend",
                "base_directory": base_directory,
            },
        }
        site_name="my_data_docs_site"

        context.add_data_docs_site(site_name=site_name,site_config=site_config)

        context.build_data_docs(site_names=site_name)
        
        context.open_data_docs()

        return True