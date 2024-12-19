from pydantic import BaseModel
from typing import Any, Dict

class ModelNameConfig(BaseModel):
    """Model Configurations"""

    model_name: str = "lightgbm"
    fine_tuning: bool = False

    class Config:
        protected_namespaces = ()

class StepConfig:
    """Configuration for pipeline steps"""
    @staticmethod
    def get_ingest_config() -> Dict[str, Any]:
        return {}
    
    @staticmethod
    def get_clean_config() -> Dict[str, Any]:
        return {}
    
    @staticmethod
    def get_train_config() -> Dict[str, Any]:
        return {
            "model_name": "lightgbm",
            "fine_tuning": False
        }
    
    @staticmethod
    def get_evaluation_config() -> Dict[str, Any]:
        return {}
