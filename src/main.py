from InquirerPy.base.control import Choice
from PyQt6.QtWidgets import QApplication
from typer import Typer

from .gui import MainWindow
from .trainer import trainer_cli


app_cli = Typer(rich_markup_mode="rich")

app_cli.add_typer(trainer_cli, name="train")


@app_cli.command()
def start():
    """
    Starts the training process.
    """
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec()


@app_cli.command()
def run():
    choices = [
        Choice(
            value="1", name="Taux d'apprentissage (alpha) \U0001f4d6", default="0.1"
        ),
        Choice(value="2", name="Taux de réduction (gamma) \U0001f4d6", default="0.6"),
        Choice(
            value="3",
            name="Taux d'exploration initial (epsilon_initial) \U0001f4d6",
            default="1.0",
        ),
        Choice(
            value="4",
            name="Taux de décroissance de l'exploration (epsilon_decay) \U0001f4d6",
            default="0.995",
        ),
        Choice(
            value="5",
            name="Taux d'exploration final (epsilon_final) \U0001f4d6",
            default="0.01",
        ),
    ]
