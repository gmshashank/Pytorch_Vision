import os
import numpy as np
import random
import torch
import torch.nn as nn
import yaml

from typing import Any, List, Tuple, Dict
from types import ModuleType
from utils.logger import setup_logger

logger = setup_logger(__name__)


def get_instance(module: ModuleType, name: str, config: Dict, *args: Any) -> Any:
    # create instance from constructor name and module name
    constr_name = config[name]["type"]
    logger.info(f"Building: {module.__name__}.{constr_name}")
    return getattr(module, constr_name)(*args, **config[name]["args"])


def load_config(file_name: str) -> dict:
    # Loading a configuration YAML file
    with open(file_name) as file_handle:
        config = yaml.safe_load(file_handle)

    return config


def setup_device(
    model: nn.Module, target_device: str
) -> Tuple[torch.device, List[int]]:

    device = torch.device(f"{target_device}")
    try:
        model = model.to(device)
    except:
        device = torch.device("cpu")
        model = model.to(device)

    return model, device


def setup_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def setup_model_params(model: nn.Module, config: Dict) -> List:
    return [{"params": model.parameters(), **config}]
