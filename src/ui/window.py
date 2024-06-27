"""
This module contains the main window class for the gui application.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget


class MainWindow(QWidget):
    """
    A class representing the main window of the RL Taxi-v3 training monitor GUI.

    The MainWindow class initializes the GUI elements for starting and stopping training, and setting hyperparameters.
    """

    def __init__(self) -> None:
        super().__init__()

        self.colorScheme = Qt.ColorScheme.Dark
        # Create central widget and set it to the main window
        # self.central_widget = QWidget(self)
        # self.central_widget.setAlignment(Qt.AlignCenter)
        # self.central_widget.setStyleSheet("""
        #     background-color: #262626;
        #     color: #FFFFFF;
        #     font-family: Titillium;
        #     font-size: 18px;
        #     """)
        # self.setCentralWidget(self.central_widget)

        self.setCursor(Qt.CursorShape.LastCursor)

        self.current_palette = self.palette()
        self.current_palette.setColorGroup(
            self.current_palette.ColorGroup.All,
            Qt.GlobalColor.white,
            Qt.GlobalColor.darkGray,
            Qt.GlobalColor.darkGray,
            Qt.GlobalColor.darkGray,
            Qt.GlobalColor.darkGray,
            Qt.GlobalColor.darkGray,
            Qt.GlobalColor.darkGray,
            Qt.GlobalColor.darkGray,
            Qt.GlobalColor.black,
        )

        self.setPalette(self.current_palette)

        # self.showMaximized()
