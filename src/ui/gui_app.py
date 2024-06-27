"""_summary_"""

from PyQt6.QtWidgets import QApplication


class GUIApp(QApplication):
    def __init__(self, *args) -> None:  # type: ignore
        """_summary_"""
        super().__init__([*args])

    def run(self) -> int:
        """_summary_"""
        return self.exec()
