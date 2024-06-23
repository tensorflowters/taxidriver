from typing import Any

import gymnasium as gym
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty, pprint
from typer import Typer

trainer_cli = Typer(rich_markup_mode="rich")

# For plotting metrics
all_epochs = []  # type: ignore
all_penalties = []  # type: ignore

console = Console(color_system="truecolor", force_terminal=True)


def pt(title: str, value: Any):
    pprint(title, expand_all=True)
    pprint(value, expand_all=True)
    print("\n")


def pt_panel(title: str, value: Any):
    pretty = Pretty(value)
    panel = Panel(pretty, title=title)
    console.print(panel)


@trainer_cli.command()
def train():
    env = gym.make("Taxi-v3", render_mode="human")  # type: ignore

    pt("Spec", env.spec)
    pt("Meta", env.metadata)
    pt("Render Mode", env.render_mode)
    pt("Observations", env.observation_space)
    """
        **States**:
            There are 500 discrete states since there are:
            - 25 taxi positions
            - 5 possible locations of the passenger (including the case when the passenger is in the taxi)
            - 4 destination locations.
        **Destination**:
            Represented on the map with the first letter of the color.
        **Passenger locations**:
            0: Red
            1: Green
            2: Yellow
            3: Blue
            4: In taxi
        **Destinations**:
            0: Red
            1: Green
            2: Yellow
            3: Blue
        **Notes**:
            An observation is returned as an int() that encodes the corresponding state,
            calculated by ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination

            Note that there are 400 states that can actually be reached during an episode.

            The missing states correspond to situations in which the passenger is at the same location as their destination, as this typically signals the end of an episode.

            Four additional states can be observed right after a successful episodes, when both the passenger and the taxi are at the destination.

            This gives a total of 404 reachable discrete states.
    """
    pt("Actions", env.action_space)
    """
        **Actions**:
            The action shape is (1,) in the range {0, 5} indicating which direction to move the taxi or to pickup/drop off passengers.
                0: Move south (down)
                1: Move north (up)
                2: Move east (right)
                3: Move west (left)
                4: Pickup passenger
                5: Drop off passenger
    """

    # Initialize Q-table
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    pt("Q-table", q_table)

    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 1.0  # Start with a high epsilon
    epsilon_min = 0.1
    epsilon_decay = 0.995

    for _ in range(1, 2):
        done = False
        observation, info = env.reset()
        pt("Is terminated or trunkated", done)
        pt_panel("Observations", observation)
        pt_panel("Infos", info)

        while not done:
            """
                As taxi is not stochastic, the transition probability is always 1.0. Implementing a transitional probability in line with the Dietterich paper (‘The fickle taxi task’) is a TODO.

                For some cases, taking an action will have no effect on the state of the episode. In v0.25.0, info["action_mask"] contains a np.ndarray for each of the actions specifying if the action will change the state.
            """
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample(info["action_mask"])
            else:
                """
                    This code is using the NumPy library to find the index of the maximum value in a specific part of a table (`q_table`) based on some conditions.

                    Here's what each part does:

                    - `np.argmax`: This function returns the indices of the maximum values along a specified axis. In this case, since no axis is specified, it will return the indices of all maximum values.

                    - `q_ table[observation, ...]`: This is selecting a specific row from the `q_table`. The `observation` variable seems to be an index or key that is used to select a particular row in the table.

                    - `np.where(info["action_mask"] == 1)[0]`: This is finding the indices of all values in the `info["action_mask"]` array where the value is equal to 1. The `[0]` at the end means that it will only return the first dimension of the result, which would be a 1D array of indices.

                    - So, when you combine these two parts, `q_ table[observation, ...]` with the indices from `np.where`, it's like saying: "Find the maximum value in the specified row (`observation`) of `q_table` where the action mask is equal to 1".

                    In other words, this code is selecting the best possible action for a given observation based on the Q-table and some condition represented by the `info["action_mask"]`.

                    For example, imagine you have a Q-table that represents the value of taking different actions in a game. The Q-table would be a 2D array where each row corresponds to a specific state or observation in the game, and each column corresponds to an action.
                """
                action = np.argmax(
                    q_table[observation, np.where(info["action_mask"] == 1)[0]]
                )

            observation, reward, terminated, truncated, info = env.step(action)

            pt_panel("Episode - Observation", observation)

            pt_panel("Reward - Observation", reward)

            pt_panel("Terminated - Observation", terminated)

            pt_panel("Truncated - Observation", truncated)

            pt_panel("Infos - Observation", info)

            # old_value = q_table[state, action]
            # next_max = np.max(q_table[next_state])

            # new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            # q_table[state, action] = new_value

            # if reward == -10:
            #     penalties += 1

            # state = next_state
            # epochs += 1

        # # Decay epsilon
        # if epsilon > epsilon_min:
        #     epsilon *= epsilon_decay

        # if i % 100 == 0:
        #     clear_output(wait=True)
        #     print(f"Episode: {i}")
