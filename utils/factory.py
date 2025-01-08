from models.base import BaseLearner
import importlib

def get_model(model_name: str, args: dict) -> BaseLearner:
    model_name = model_name.lower()
    
    try:
        # Dynamically load model's class
        module = importlib.import_module(f"models.{model_name}")
        model_class = getattr(module, "Learner")
    except (ModuleNotFoundError, AttributeError):
        raise ValueError(f"Model {model_name} not found.")
    
    return model_class(args)