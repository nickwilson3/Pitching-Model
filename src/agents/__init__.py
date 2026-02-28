from .data_scientist import DataScientistAgent
from .data_engineer import DataEngineerAgent
from .data_analyst import DataAnalystAgent

AGENTS = {
    "scientist": DataScientistAgent,
    "engineer":  DataEngineerAgent,
    "analyst":   DataAnalystAgent,
}

__all__ = ["DataScientistAgent", "DataEngineerAgent", "DataAnalystAgent", "AGENTS"]
