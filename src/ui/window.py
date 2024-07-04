"""
This module contains the main window class for the gui application.
"""

import random
from typing import Any

import matplotlib  # import matplotlib after PyQt6
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    # QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from ..config.base import Config

matplotlib.use("QtAgg")


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):  # type: ignore
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)  # type: ignore


class MainWindow(QMainWindow):
    config = Config()
    config_data: dict = {}
    dark_mode: bool = True
    darkPalette: QPalette = QPalette()
    plots: list[dict[Any, Any]]

    def __init__(self) -> None:
        super().__init__()

        self.config_data = self.config.get_dict()

        print(self.config_data)

        self.dark_mode: bool = True
        self.darkPalette: QPalette = QPalette()

        self.plots = [
            (
                dict(
                    **x,
                    sc=MplCanvas(self, width=5, height=4, dpi=100),  # type: ignore
                    df=pd.DataFrame(
                        data=[
                            [0, 10],
                            [5, 15],
                            [2, 20],
                            [15, 25],
                            [4, 10],
                        ],
                        columns=x["cols"],
                    ),
                )
            )
            for x in self.config_data["plots"]
        ]

        self.setMinimumSize(QSize(1800, 1000))

    def setup_dark_mode(self) -> None:
        self.darkPalette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        self.darkPalette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        self.darkPalette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.WindowText,
            QColor(127, 127, 127),
        )
        self.darkPalette.setColor(QPalette.ColorRole.Base, QColor(42, 42, 42))
        self.darkPalette.setColor(QPalette.ColorRole.AlternateBase, QColor(66, 66, 66))
        self.darkPalette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        self.darkPalette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        self.darkPalette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        self.darkPalette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.Text,
            QColor(127, 127, 127),
        )
        self.darkPalette.setColor(QPalette.ColorRole.Dark, QColor(35, 35, 35))
        self.darkPalette.setColor(QPalette.ColorRole.Shadow, QColor(20, 20, 20))
        self.darkPalette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        self.darkPalette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        self.darkPalette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.ButtonText,
            QColor(127, 127, 127),
        )
        self.darkPalette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        self.darkPalette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        self.darkPalette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        self.darkPalette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.Highlight,
            QColor(80, 80, 80),
        )
        self.darkPalette.setColor(
            QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white
        )
        self.darkPalette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.HighlightedText,
            QColor(127, 127, 127),
        )

    def setup_ui(self, disable_dark_mode: bool = False) -> None:
        if disable_dark_mode:
            self.dark_mode = False
        else:
            self.setup_dark_mode()
            self.setup_dark_palette()

        page_layout = QVBoxLayout()
        self.graph_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        page_layout.addLayout(self.graph_layout)
        page_layout.addLayout(button_layout)

        for w in self.config_data["widgets"]:
            if w["type"] == "button":
                btn = QPushButton(w["label"])
                if w["id"] == "start_btn":
                    btn.clicked.connect(self.start_agent)
                btn.setMinimumHeight(40)
                button_layout.addWidget(btn)

        widget = QWidget()
        widget.setLayout(page_layout)

        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        self.setCentralWidget(widget)

    def setup_dark_palette(self) -> None:
        self.setStyleSheet("color: white")

    def set_plot_widget(self) -> None:
        for p in self.plots:
            p["df"].plot(ax=p["sc"].axes)
            self.graph_layout.addWidget(p["sc"])

    def update_plot_widget(self, plot_id: str, data: list[list[Any]]) -> None:
        for p in self.plots:
            if plot_id == p["id"]:
                p["sc"].axes.clear()
                p["df"] = pd.DataFrame(
                    data=data,
                    columns=p["cols"],
                )
                p["df"].plot(ax=p["sc"].axes)
                p["sc"].draw()

    def start_agent(self) -> None:
        self.update_plot_widget(
            ["reward_plot", "epsilon_plot"][random.randint(0, 1)],
            [
                [0, random.randint(5, 30)],
                [1, random.randint(5, 30)],
                [2, random.randint(5, 30)],
                [3, random.randint(5, 30)],
                [4, random.randint(5, 30)],
            ],
        )
