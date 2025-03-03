from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataValidationConfig:
    data_source_path_kaggle: str
    data_source_path_gdrive: str
    Status_report: str
    all_schema: dict
    
    