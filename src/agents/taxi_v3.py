"""
Q-Learning agent aiming to solve the Taxi-v3 environment.
"""

import gymnasium as gym
import numpy as np
from gymnasium.spaces.discrete import Discrete

from ..settings.logger import p as prt


class TaxiV3Agent:
    def __init__(
        self,
        alpha: float,  # Learning rate
        epsilon: float,  # Exploration rate initial
        epsilon_decay: float,  # Exploration rate decay
        epsilon_minimal: float,  # Exploration rate final
        gamma: float = 0.95,  # Discount factor
    ) -> None:
        self.__env_name__ = "Taxi-v3"

        self.env = gym.make(self.__env_name__)

        self.observation_space: gym.spaces.Space[Discrete] = self.env.observation_space
        self.action_space: gym.spaces.Space[Discrete] = self.env.action_space

        self.__obs_space_shape__ = (
            self.observation_space.n if hasattr(self.observation_space, "n") else None
        )
        self.__act_space_shape__ = (
            self.action_space.n if hasattr(self.action_space, "n") else None
        )

        self.q_values: np.ndarray = np.zeros(
            (self.__obs_space_shape__, self.__act_space_shape__),
            dtype=np.uint8,
        )

        prt(self.q_values, title="Q-values table")
        prt(self.action_space.sample(), title="Action space")
        prt(
            self.observation_space.sample(),
            title="Observation space",
        )

        # Hyperparameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_minimal = epsilon_minimal
        self.gamma = gamma

        self.training_error: list = []

    def get_action(
        self, observation: int, info: dict[str, np.ndarray | float | int]
    ) -> int:
        """
        **Summary**:
          This code is using the NumPy library to find the index of the maximum value in a specific part of a table (`q_table`) based on some conditions.
          Here's what each part does:
            - `np.argmax`: This function returns the indices of the maximum values along a specified axis. In this case, since no axis is specified, it will return the indices of all maximum values.
            - `q_ table[observation, ...]`: This is selecting the `observation` row from the `q_table` to update.
            - `np.where(info["action_mask"] == 1)[0]`: This is finding the indices of all values in the `info["action_mask"]` array where the value is equal to 1. The `[0]` at the end means that it will only return the first dimension of the result, which would be a 1D array of indices.

            In other words, this code is selecting the best possible action for a given observation based on the Q-table and some condition represented by the `info["action_mask"]`.
        **Args**:
            state: The current state
        **Returns**:
            As taxi is not stochastic, the transition probability is always 1.0.
            Implementing a transitional probability in line with the Dietterich paper (‘The fickle taxi task’) is a TODO.
            For some cases, taking an action will have no effect on the state of the episode. In v0.25.0, info["action_mask"] contains a np.ndarray for each of the actions specifying if the action will change the state.
        """
        if np.random.random() < self.epsilon:
            return int(self.env.action_space.sample(info["action_mask"]))
        else:
            return int(
                np.argmax(
                    self.q_values[observation, np.where(info["action_mask"] == 1)[0]]
                )
            )

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ) -> None:
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.gamma * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.alpha * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_minimal, self.epsilon - self.epsilon_decay)
