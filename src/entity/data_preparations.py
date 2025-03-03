from dataclasses import dataclass


@dataclass
class DataPreparationConfig:
    kaggle: str
    gdrive: str
    models_path: str
    processed: str
