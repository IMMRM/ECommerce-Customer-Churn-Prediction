from dataclasses import dataclass

@dataclass
class DataStorageConnectionConfig:
    db_path: str
    processed: str
    data_table_name: str