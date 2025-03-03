
from dataclasses import dataclass

@dataclass
class CustomerData:
    customer_id: int
    churn_flag: int
    tenure: int
    preferred_device: str
    city_tier: int
    warehouse_to_home: float
    preferred_payment: str
    gender: str
    hours_spent: float
    num_devices: int
    preferred_order_category: str
    satisfaction_score: int
    marital_status: str
    num_addresses: int
    complaints: int
    order_amount_hike: float
    coupons_used: int
    order_count: int
    days_since_last_order: int
    cashback_amount: float
    source: str
    last_updated_date: str

@dataclass
class DataIngestionConfig:
    kaggle_dataset: str
    gdrive_dataset: str
    local_dataset: str
    staging_dataset: str
    staging_kaggle: str
    staging_gdrive: str
    
    
    
    

    