"""
**Summary**:
    Will end up only responsible for GUI elements and interactions.
    The training logic will be moved to the `TaxiV3` class.
    For now it's holding all the application logic.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    """
    A class representing the main window of the RL Taxi-v3 training monitor GUI.

    The MainWindow class initializes the GUI elements for starting and stopping training, and setting hyperparameters.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setup_ui()
        self.setup_environment()
        self.setup_timer()
        self.setup_plots()
        self.show()

        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 1.0  # Start with a high epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        # Initialize training control variables
        self.current_episode = 0
        self.max_episodes = (
            100_000  # Define the maximum number of episodes for training
        )

        # For plotting metrics
        self.all_epochs: list = []
        self.all_penalties: list = []

        self.timer = None

    def setup_ui(self) -> None:
        # Initialize UI elements
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)  # type: ignore
        self.setup_controls()

    def setup_environment(self) -> None:
        # Initialize the Gym environment
        self.env = gym.make("Taxi-v3")
        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])  # type: ignore

    def setup_timer(self) -> None:
        # Timer for training updates
        self.timer = QTimer()  # type: ignore
        self.timer.setInterval(50)  # type: ignore  # Update interval in milliseconds
        self.timer.timeout.connect(self.update_training)  # type: ignore

    def setup_plots(self) -> None:
        # Initialize plotting widget
        self.graphWidget = pg.PlotWidget()
        self.layout.addWidget(self.graphWidget)  # type: ignore
        self.epoch_plot = self.graphWidget.plot([], [], pen="y")
        self.penalty_plot = self.graphWidget.plot([], [], pen="r")

    def setup_controls(self) -> None:
        # Control buttons and hyperparameter inputs
        self.alpha_input = QLineEdit("0.1")
        self.gamma_input = QLineEdit("0.6")
        self.epsilon_input = QLineEdit("1.0")
        self.add_parameter_controls()
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        self.layout.addWidget(self.start_button)  # type: ignore
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.layout.addWidget(self.stop_button)  # type: ignore
        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        self.layout.addWidget(self.save_button)  # type: ignore

    def add_parameter_controls(self) -> None:
        self.layout.addWidget(QLabel("Alpha (Learning Rate):"))  # type: ignore
        self.layout.addWidget(self.alpha_input)  # type: ignore
        self.layout.addWidget(QLabel("Gamma (Discount Factor):"))  # type: ignore
        self.layout.addWidget(self.gamma_input)  # type: ignore
        self.layout.addWidget(QLabel("Epsilon (Exploration Rate):"))  # type: ignore
        self.layout.addWidget(self.epsilon_input)  # type: ignore

    # TODO - Separate the training logic from the GUI code
    def start_training(self) -> None:
        self.alpha = float(self.alpha_input.text())
        self.gamma = float(self.gamma_input.text())
        self.epsilon = float(self.epsilon_input.text())
        self.timer.start()  # type: ignore

    def stop_training(self) -> None:
        self.timer.stop()  # type: ignore

    def save_results(self) -> None:
        np.save("q_table.npy", self.q_table)
        print("Training data saved.")

    def update_training(self) -> None:
        observation, info = self.env.reset()
        epochs, penalties, done = 0, 0, False
        while not done:
            action = None
            if np.random.random() < self.epsilon:
                action = self.env.action_space.sample(info["action_mask"])
            else:
                action = np.argmax(
                    self.q_values[observation, np.where(info["action_mask"] == 1)[0]]  # type: ignore
                )
            next_state, reward, terminated, truncated, info = self.env.step(action)
            self.update_q_table(observation, action, reward, next_state)
            observation = next_state
            penalties += reward == -10
            epochs += 1
            done = terminated or truncated
        self.post_episode_update(epochs, penalties)

    def update_q_table(
        self, state: Any, action: Any, reward: Any, next_state: Any
    ) -> None:
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (
            reward + self.gamma * next_max
        )
        self.q_table[state, action] = new_value

    def post_episode_update(self, epochs: Any, penalties: Any) -> None:
        self.all_epochs.append(epochs)
        self.all_penalties.append(penalties)
        self.epoch_plot.setData(self.all_epochs)
        self.penalty_plot.setData(self.all_penalties)
        self.current_episode += 1
        if self.current_episode >= self.max_episodes:
            self.timer.stop()  # type: ignore
            print("Training finished.")
