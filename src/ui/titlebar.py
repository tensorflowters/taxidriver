import os

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QCursor, QIcon
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QToolButton,
    QWidget,
)


class CustomTitleBar(QWidget):
    def __init__(self, parent):  # type: ignore
        super().__init__(parent)
        self.initial_pos = None
        title_bar_layout = QHBoxLayout(self)
        title_bar_layout.setContentsMargins(1, 0, 1, 0)
        title_bar_layout.setSpacing(0)
        self.title = QLabel(f"{self.__class__.__name__}", self)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet("""
        QLabel { text-transform: uppercase; font-size: 18pt; margin-left: 48px; font-weight: bold;}
        """)

        if title := parent.windowTitle():
            self.title.setText(title)
        title_bar_layout.addWidget(self.title)

        basedir = os.path.dirname(__file__)

        # Min button
        self.min_button = QToolButton(self)
        min_icon = QIcon()
        min_icon.addFile(os.path.join(basedir, "min.svg"))
        self.min_button.setIcon(min_icon)
        self.min_button.setIconSize(QSize(32, 32))
        self.min_button.clicked.connect(self.window().showMinimized)  # type: ignore

        # Max button
        self.max_button = QToolButton(self)
        max_icon = QIcon()
        max_icon.addFile(os.path.join(basedir, "max.svg"))
        self.max_button.setIcon(max_icon)
        self.max_button.setIconSize(QSize(32, 32))
        self.max_button.clicked.connect(self.window().showMaximized)  # type: ignore

        # Close button
        self.close_button = QToolButton(self)
        close_icon = QIcon()
        close_icon.addFile(
            os.path.join(basedir, "close.svg")
        )  # Close has only a single state.
        self.close_button.setIconSize(QSize(32, 32))
        self.close_button.setIcon(close_icon)
        self.close_button.clicked.connect(self.window().close)  # type: ignore

        # Normal button
        self.normal_button = QToolButton(self)
        normal_icon = QIcon()
        normal_icon.addFile(os.path.join(basedir, "normal.svg"))
        self.normal_button.setIconSize(QSize(32, 32))
        self.normal_button.setIcon(normal_icon)
        self.normal_button.clicked.connect(self.window().showNormal)  # type: ignore
        self.normal_button.setVisible(False)
        # Add buttons
        buttons = [
            self.min_button,
            self.normal_button,
            self.max_button,
            self.close_button,
        ]
        for button in buttons:
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setFixedSize(QSize(40, 40))
            button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            button.setStyleSheet(
                """QToolButton {
                    border: none;
                    padding-left: 4px;
                    padding-right: 0;
                    padding-top: 0;
                    padding-bottom: 0;
                }
                """
            )
            title_bar_layout.addWidget(button)

    def window_state_changed(self, state):  # type: ignore
        if state == Qt.WindowState.WindowMaximized:
            self.normal_button.setVisible(True)
            self.max_button.setVisible(False)
        else:
            self.normal_button.setVisible(False)
            self.max_button.setVisible(True)
