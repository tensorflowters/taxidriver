"""
Q-Learning agent aiming to solve the Taxi-v3 environment.
"""

from collections import defaultdict
from typing import DefaultDict

import gymnasium as gym
import numpy as np

env = gym.make("Taxi-v3", render_mode="human")


class TaxiV3Agent:
    def __init__(
        self,
        alpha: float,  # Learning rate
        epsilon_initial: float,  # Exploration rate initial
        epsilon_decay: float,  # Exploration rate decay
        epsilon_final: float,  # Exploration rate final
        gamma: float = 0.95,  # Discount factor
    ) -> None:
        """
        **Summary**:
          Initialize a Reinforcement Learning agent with an empty dictionary
          of state-action values (q-tabble), a learning rate, exploration rate
          and discount factor.

        **Args**:
            alpha: The learning rate
            epsilon_initial: The initial epsilon value
            epsilon_decay: The decay for epsilon
            epsilon_final: The final epsilon value
            gamma: The discount factor for computing the Q-value
        """
        self.q_values: DefaultDict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(env.action_space.n)
        )

        self.alpha = alpha

        self.epsilon_initial = epsilon_initial
        self.epsilon_decay = epsilon_decay
        self.epsilon_final = epsilon_final

        self.gamma = gamma

        self.training_error = []

    def get_action(self, observation, info) -> int:
        """
        **Summary**:
          Get the action to take based on the epsilon-greedy policy.
          This code is using the NumPy library to find the index of the maximum value in a specific part of a table (`q_table`) based on some conditions.
          Here's what each part does:
            - `np.argmax`: This function returns the indices of the maximum values along a specified axis. In this case, since no axis is specified, it will return the indices of all maximum values.
            - `q_ table[observation, ...]`: This is selecting a specific row from the `q_table`. The `observation` variable seems to be an index or key that is used to select a particular row in the table.
            - `np.where(info["action_mask"] == 1)[0]`: This is finding the indices of all values in the `info["action_mask"]` array where the value is equal to 1. The `[0]` at the end means that it will only return the first dimension of the result, which would be a 1D array of indices.
            - So, when you combine these two parts, `q_ table[observation, ...]` with the indices from `np.where`, it's like saying: "Find the maximum value in the specified row (`observation`) of `q_table` where the action mask is equal to 1".
            In other words, this code is selecting the best possible action for a given observation based on the Q-table and some condition represented by the `info["action_mask"]`.
            For example, imagine you have a Q-table that represents the value of taking different actions in a game. The Q-table would be a 2D array where each row corresponds to a specific state or observation in the game, and each column corresponds to an action.
        **Args**:
            state: The current state
        **Returns**:
            As taxi is not stochastic, the transition probability is always 1.0.
            Implementing a transitional probability in line with the Dietterich paper (‘The fickle taxi task’) is a TODO.
            For some cases, taking an action will have no effect on the state of the episode. In v0.25.0, info["action_mask"] contains a np.ndarray for each of the actions specifying if the action will change the state.
        """
        if np.random.random() < self.epsilon_initial:
            return env.action_space.sample(info["action_mask"])
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
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.gamma * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.alpha * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon_initial = max(
            self.final_epsilon, self.epsilon_initial - self.epsilon_decay
        )
