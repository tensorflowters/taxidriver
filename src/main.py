import sys

import gymnasium as gym
from InquirerPy.base.control import Choice
from PyQt6.QtWidgets import QApplication
from typer import Typer

from .agents.taxi_v3 import TaxiV3Agent
from .cli.prompts import default_prompt_float
from .config.base import HyperparametersConfig
from .settings.logger import p as print
from .ui.window import MainWindow

taxi_cli = Typer(
    add_completion=False,
    name="taxi",
    no_args_is_help=True,
    rich_help_panel="rich",
    rich_markup_mode="rich",
)


@taxi_cli.command("run_agent", help="Run the agent in the Taxi-v3 environment.")
def run_agent() -> None:
    print(gym.envs.registry.keys(), "Environment registery", "All envs keys")

    _ = TaxiV3Agent(*HyperparametersConfig())


# @taxi_cli.command(
#     "run_app_debug",
#     help="Run the GUI application. This is temporary and will not include holding all the app logic.",
# )
# def run_app_debug() -> None:
#     app = QApplication([])
#     main_window = MainDebugWindow()
#     main_window.show()
#     app.exec()


@taxi_cli.command(
    "run_app",
    help="Run the GUI application. This is temporary and will not include holding all the app logic.",
)
def run_app() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    win = MainWindow()

    app.setApplicationDisplayName("Taxi-v3")
    app.setApplicationName("Taxi-v3")
    app.setApplicationVersion("0.1.0")

    win.show()

    app.exec()


@taxi_cli.command(
    "init_train",
    help="How you would actually start the training. Asking for hyperparameters.",
)
def init_train() -> None:
    choices = [
        (
            Choice(value=0.1, name="Taux d'apprentissage (alpha): "),
            dict(id="alpha", default=0.1),
        ),
        (
            Choice(value=0.9, name="Facteur de réduction (gamma): "),
            dict(id="gamma", default=0.9),
        ),
        (
            Choice(value=0.1, name="Probabilité d'exploration (epsilon): "),
            dict(id="epsilon", default=0.1),
        ),
        (
            Choice(value=0.995, name="Décroissance de epsilon (epsilon_decay): "),
            dict(id="epsilon_decay", default=0.995),
        ),
        (
            Choice(value=0.01, name="Epsilon minimal (epsilon_minimal): "),
            dict(id="epsilon_minimal", default=0.01),
        ),
    ]

    user_choices = [
        dict(
            id=meta.get("id"),
            value=default_prompt_float(choice.name, meta.get("default")),
        )
        for choice, meta in choices
    ]

    print(
        user_choices,
        title="User choices",
        msg="Should see alpha, gamma, epsilon, etc...",
    )
