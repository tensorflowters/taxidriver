"""
This module contains the main window class for the gui application.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QMainWindow,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.dark_mode: bool = True

    def setup_dark_mode(self):
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

    def setup_ui(self, disable_dark_mode: bool = False):
        if disable_dark_mode:
            self.dark_mode = False
        else:
            self.darkPalette = QPalette()
            self.self.setup_dark_palette()

        # layout = QVBoxLayout()
        # widgets = [
        #     QCheckBox,
        #     QComboBox,
        #     QDateEdit,
        #     QDateTimeEdit,
        #     QDial,
        #     QDoubleSpinBox,
        #     QFontComboBox,
        #     QLCDNumber,
        #     QLabel,
        #     QLineEdit,
        #     QProgressBar,
        #     QPushButton,
        #     QRadioButton,
        #     QSlider,
        #     QSpinBox,
        #     QTimeEdit,
        # ]

        # for w in widgets:
        #     layout.addWidget(w())

        # widget = QWidget()
        # widget.setLayout(layout)

        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        # self.setCentralWidget(widget)

    def setup_dark_palette(self):
        self.setStyleSheet("color: white")
