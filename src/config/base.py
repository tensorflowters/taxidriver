from pathlib import Path

from confz import BaseConfig, FileSource


class HyperparametersConfig(BaseConfig):  # type: ignore
    alpha: float
    epsilon: float
    epsilon_decay: float
    epsilon_minimal: float
    gamma: float

    CONFIG_SOURCES = FileSource(file=Path(__file__).parent / "hyperparameters.yml")
