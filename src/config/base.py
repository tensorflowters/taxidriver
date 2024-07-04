from dataclasses import dataclass
from pathlib import Path

from confz import BaseConfig, ConfigSource, FileSource


@dataclass
class ButtonUI(ConfigSource):  # type: ignore
    label: str
    id: str
    type: str


@dataclass
class Plots(ConfigSource):  # type: ignore
    type: str
    id: str
    title: str
    cols: list[str]


class Config(BaseConfig):  # type: ignore
    alpha: float
    epsilon: float
    epsilon_decay: float
    epsilon_minimal: float
    gamma: float
    widgets: list[ButtonUI]
    plots: list[Plots]

    CONFIG_SOURCES = FileSource(file=Path(__file__).parent / "config_params.yml")

    def get_dict(self) -> dict:
        return self.model_dump()
