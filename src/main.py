import gymnasium as gym
import pyqtgraph as pg
from InquirerPy.base.control import Choice
from PyQt6 import QtWidgets
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
    # app = GUIApp()
    app = QtWidgets.QApplication([])
    w = MainWindow()
    # app.setActiveWindow(w)
    # w = QtWidgets.QWidget()
    w.setWindowTitle("PyQtGraph example")

    ## Create some widgets to be placed inside
    btn = QtWidgets.QPushButton("press me")
    text = QtWidgets.QLineEdit("enter text")
    listw = QtWidgets.QListWidget()
    plot = pg.PlotWidget()

    ## Create a grid layout to manage the widgets size and position
    layout = QtWidgets.QGridLayout()
    w.setLayout(layout)

    ## Add widgets to the layout in their proper positions
    layout.addWidget(btn, 0, 0)  # button goes in upper-left
    layout.addWidget(text, 1, 0)  # text edit goes in middle-left
    layout.addWidget(listw, 2, 0)  # list widget goes in bottom-left
    layout.addWidget(plot, 0, 1, 3, 1)  # plot goes on right side, spanning 3 rows
    ## Display the widget as a new window
    w.show()
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
