import gymnasium as gym
import numpy as np
import pyqtgraph as pg  # type: ignore
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

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_environment()
        self.setup_timer()
        self.setup_plots()
        self.show()

        # Initialize training control variables
        self.current_episode = 0
        self.max_episodes = 100000  # Define the maximum number of episodes for training

        # For plotting metrics
        self.all_epochs = []
        self.all_penalties = []

    def setup_ui(self):
        # Initialize UI elements
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.setup_controls()

    def setup_environment(self):
        # Initialize the Gym environment
        self.env = gym.make("Taxi-v3").env
        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])

    def setup_timer(self):
        # Timer for training updates
        self.timer = QTimer()
        self.timer.setInterval(50)  # Update interval in milliseconds
        self.timer.timeout.connect(self.update_training)

    def setup_plots(self):
        # Initialize plotting widget
        self.graphWidget = pg.PlotWidget()
        self.layout.addWidget(self.graphWidget)
        self.epoch_plot = self.graphWidget.plot([], [], pen="y")
        self.penalty_plot = self.graphWidget.plot([], [], pen="r")

    def setup_controls(self):
        # Control buttons and hyperparameter inputs
        self.alpha_input = QLineEdit("0.1")
        self.gamma_input = QLineEdit("0.6")
        self.epsilon_input = QLineEdit("1.0")
        self.add_parameter_controls()
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        self.layout.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.layout.addWidget(self.stop_button)
        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        self.layout.addWidget(self.save_button)

    def add_parameter_controls(self):
        self.layout.addWidget(QLabel("Alpha (Learning Rate):"))
        self.layout.addWidget(self.alpha_input)
        self.layout.addWidget(QLabel("Gamma (Discount Factor):"))
        self.layout.addWidget(self.gamma_input)
        self.layout.addWidget(QLabel("Epsilon (Exploration Rate):"))
        self.layout.addWidget(self.epsilon_input)

    def start_training(self):
        self.alpha = float(self.alpha_input.text())
        self.gamma = float(self.gamma_input.text())
        self.epsilon = float(self.epsilon_input.text())
        self.timer.start()

    def stop_training(self):
        self.timer.stop()

    def save_results(self):
        np.save("q_table.npy", self.q_table)
        print("Training data saved.")

    def update_training(self):
        # Training logic integrated here with error handling for the step method
        if self.current_episode >= self.max_episodes:
            return
        state = self.env.reset()
        epochs, penalties, done = 0, 0, False
        while not done:
            action = (
                self.env.action_space.sample()
                if np.random.uniform(0, 1) < self.epsilon
                else np.argmax(self.q_table[state])
            )
            next_state, reward, done = self.env.step(action)
            self.update_q_table(state, action, reward, next_state)
            state = next_state
            penalties += reward == -10
            epochs += 1
        self.post_episode_update(epochs, penalties)

    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (
            reward + self.gamma * next_max
        )
        self.q_table[state, action] = new_value

    def post_episode_update(self, epochs, penalties):
        self.all_epochs.append(epochs)
        self.all_penalties.append(penalties)
        self.epoch_plot.setData(self.all_epochs)
        self.penalty_plot.setData(self.all_penalties)
        self.current_episode += 1
        if self.current_episode >= self.max_episodes:
            self.timer.stop()
            print("Training finished.")
