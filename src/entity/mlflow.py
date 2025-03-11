from dataclasses import dataclass

@dataclass
class MLFlowCreds:
    tracking_ui: str
    username: str
    password: str

@dataclass
class Parameters:
    data_path: str
    model_path: str
    random_state: int
    n_estimators: int
    max_depth: int
    