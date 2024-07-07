"""
This module contains the main window class for the gui application.
"""

import random
from typing import Any

import matplotlib
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QColor, QCursor, QFont, QPalette
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    # QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from ..config.base import Config
from ..settings.logger import logger
from .titlebar import CustomTitleBar

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

    def __init__(self, title: str = "Taxi-v3 Agent Training") -> None:
        super().__init__()

        self.setWindowTitle(title)
        self.resize(1800, 1100)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.config_data = self.config.get_dict()

        self.dark_mode: bool = True
        self.darkPalette: QPalette = QPalette()
        self.setFont(QFont("Consolas", 14))

        for inp in self.config_data["line_edit"]:
            setattr(
                self,
                inp["id"],
                QDoubleSpinBox(),
            )
            self.get_attr(inp["id"]).setRange(inp["min"], inp["max"])
            self.get_attr(inp["id"]).setSingleStep(inp["step"])
            self.get_attr(inp["id"]).setValue(inp["default_value"])
            self.get_attr(inp["id"]).valueChanged.connect(
                getattr(self, f"handle_{inp['id']}_changed")
            )
            self.get_attr(inp["id"]).setFont(QFont("Consolas", 14))
            self.get_attr(inp["id"]).setButtonSymbols(
                QAbstractSpinBox.ButtonSymbols.PlusMinus
            )
            self.get_attr(inp["id"]).setCursor(
                QCursor(Qt.CursorShape.PointingHandCursor)
            )
            self.get_attr(inp["id"]).setObjectName("QSpinBox")
            self.get_attr(inp["id"]).setStyleSheet("""
                #QSpinBox {
                    background-color: rgba(53, 53, 53, 0.5);
                    border-width: 2px;
                    border-radius: 10px;
                    border-color: rgb(20, 20, 20);
                    padding: 15px 25px 15px 25px;
                }
                #QSpinBox::up-button {
                    width: 36px;
                    height: 24px;
                }
                #QSpinBox::down-button {
                    width: 36px;
                    height: 24px;
                }
            """)

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

    def setup_dark_mode(self) -> None:
        self.darkPalette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        self.darkPalette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        self.darkPalette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.WindowText,
            QColor(127, 127, 127),
        )
        self.darkPalette.setColor(QPalette.ColorRole.Base, QColor(40, 40, 40))
        self.darkPalette.setColor(QPalette.ColorRole.AlternateBase, QColor(65, 65, 65))
        self.darkPalette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        self.darkPalette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        self.darkPalette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
        self.darkPalette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.Text,
            QColor(127, 127, 127),
        )
        self.darkPalette.setColor(QPalette.ColorRole.Dark, QColor(20, 20, 20))
        self.darkPalette.setColor(QPalette.ColorRole.Shadow, QColor(15, 15, 15))
        self.darkPalette.setColor(QPalette.ColorRole.Button, QColor(55, 55, 55))
        self.darkPalette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
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
            QPalette.ColorRole.HighlightedText, QColor(255, 255, 255)
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

        self.title_bar = CustomTitleBar(self)  # type: ignore
        page_layout = QVBoxLayout()
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        page_layout.addWidget(self.title_bar)

        self.graph_layout = QHBoxLayout()
        button_layout = QHBoxLayout()
        self.hyparameters_grid_layout = QFormLayout()

        page_layout.addLayout(self.hyparameters_grid_layout)
        page_layout.addLayout(self.graph_layout)
        page_layout.addLayout(button_layout)
        page_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        page_layout.setContentsMargins(40, 10, 40, 10)
        page_layout.setSpacing(30)

        for w in self.config_data["widgets"]:
            if w["type"] == "button":
                btn = QPushButton(w["label"])
                if w["id"] == "start_btn":
                    btn.clicked.connect(self.start_agent)
                btn.setMinimumHeight(50)
                btn.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
                btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                btn.setStyleSheet("""
                    background-color: rgba(53, 53, 53, 0.60);
                    border-width: 2px;
                    border-radius: 10px;
                    border-color: rgb(20, 20, 20);
                """)
                button_layout.addWidget(btn)

        for inpt in self.config_data["line_edit"]:
            label = QLabel(inpt["label"])
            label.setMinimumHeight(40)
            label.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
            self.hyparameters_grid_layout.addRow(label, self.get_attr(inpt["id"]))

        widget = QWidget()
        widget.setObjectName("Container")
        widget.setStyleSheet("""#Container {
            background: qlineargradient(x1:0 y1:0, x2:1 y2:1, stop:0 #FA080E1F stop:1 #171717);
            border-radius: 10px;
            padding-bottom: 20px;
        }""")
        widget.setLayout(page_layout)

        self.setCentralWidget(widget)

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

    def get_attr(self, attr_name: str, default: Any = None) -> Any:
        return getattr(self, attr_name, default)

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

    def handle_alpha_changed(self, val: str | float) -> float:
        val_rounded = val

        if val and isinstance(val, float):
            val_rounded = round(val, 2)
        elif val and isinstance(val, str):
            val_rounded = float(val)
            val_rounded = round(val_rounded, 2)
        else:
            logger(
                f" is not a float either a string: {val}",
                "Exploratory rate (alpha value)",
            )
            val_rounded = round(
                float(self.config_data["line_edit"][0]["default_value"]), 2
            )

        logger(f"Alpha value: {val_rounded}")
        return val_rounded

    def handle_epsilon_changed(self, val: str | float) -> None:
        print(val)

    def handle_gamma_changed(self, val: str | float) -> None:
        print(val)

    def handle_epsilon_decay_changed(self, val: str | float) -> None:
        print(val)

    def handle_epsilon_minimal_changed(self, val: str | float) -> None:
        print(val)

    def changeEvent(self, event):  # type: ignore
        if event.type() == QEvent.Type.WindowStateChange:
            self.title_bar.window_state_changed(self.windowState())  # type: ignore
        super().changeEvent(event)
        event.accept()

    def window_state_changed(self, state):  # type: ignore
        self.normal_button.setVisible(state == Qt.WindowState.WindowMaximized)  # type: ignore
        self.max_button.setVisible(state != Qt.WindowState.WindowMaximized)  # type: ignore

    def mousePressEvent(self, event):  # type: ignore
        if event.button() == Qt.MouseButton.LeftButton:
            self.initial_pos = event.position().toPoint()
        super().mousePressEvent(event)
        event.accept()

    def mouseMoveEvent(self, event):  # type: ignore
        if self.initial_pos is not None:
            delta = event.position().toPoint() - self.initial_pos
            self.window().move(  # type: ignore
                self.window().x() + delta.x(),  # type: ignore
                self.window().y() + delta.y(),  # type: ignore
            )
        super().mouseMoveEvent(event)
        event.accept()

    def mouseReleaseEvent(self, event):  # type: ignore
        self.initial_pos = None
        super().mouseReleaseEvent(event)
        event.accept()
