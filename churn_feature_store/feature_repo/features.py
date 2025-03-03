# checking feature store
from feast import (Feature, Entity, FeatureView, ValueType,FileSource,Field,PushSource)
from feast.types import Int64,Float64
from pathlib import Path

#Define the entity (Customer)
customer=Entity(
    name="CustomerID",
    value_type=ValueType.INT64,
    description="Customer ID",
    join_keys=["CustomerID"]
)

# #Define the data source
data_source=FileSource(
    path=str(Path(__file__).parent.parent.parent/"data/processed/transformed_churn_data.parquet"),
    timestamp_field="event_timestamp"
    
)

customer_features=FeatureView(
    name="customer_features", #Name of this feature view
    entities=[customer],
    ttl=None,
    schema=[
        # Field(name="Tenure", dtype=Float64),
        # Field(name="NumberOfDeviceRegistered",dtype=Int64),
        # Field(name="PreferedOrderCat",dtype=Int64),
        # Field(name="SatisfactionScore",dtype=Int64),
        # Field(name="MaritalStatus",dtype=Int64),
        Field(name="TotalSpend",dtype=Float64),
        Field(name="ActivityFrequency",dtype=Float64)
        
    ],
    online=True,
    source=data_source,
)
# Defines a way to push data (to be available offline, online or both) into Feast.
data_source_sqlite = PushSource(
    name="data_source_sqlite",
    batch_source=data_source
)