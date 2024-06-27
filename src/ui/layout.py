"""
Layout module for the UI.
"""

from PyQt6.QtWidgets import QGridLayout


class Layout(QGridLayout):
    def __init__(self) -> None:
        super().__init__()
        self.setEnabled(True)
