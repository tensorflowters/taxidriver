from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from confz import BaseConfig, ConfigSource, FileSource


@dataclass
class ButtonUI(ConfigSource):
    label: str
    id: str
    type: str


@dataclass
class Plots(ConfigSource):
    type: str
    id: str
    title: str
    cols: list[str]


@dataclass
class Input(ConfigSource):
    type: str
    id: Literal["alpha", "gamma", "epsilon", "epsilon_decay", "epsilon_minimal"]
    label: str
    default_value: float
    step: float
    max: float
    min: float


class Config(BaseConfig):  # type: ignore
    line_edit: list[Input]
    widgets: list[ButtonUI]
    plots: list[Plots]

    CONFIG_SOURCES = FileSource(file=Path(__file__).parent / "config_params.yml")

    def get_dict(self) -> dict:
        return self.model_dump()
