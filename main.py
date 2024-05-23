import os
import tkinter as tk
from typing import List

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import gymnasium as gym

class RLGuiApp:
    root: tk.Tk
    Q_: np.ndarray | None
    success_rate: float | None
    left_frame: tk.Frame
    right_frame: tk.Frame
    env_var: tk.StringVar
    num_episodes_var: tk.IntVar
    lr_var: tk.DoubleVar
    gamma_var: tk.DoubleVar
    num_test_episodes_var: tk.IntVar
    train_fig: plt.Figure
    train_ax: plt.Axes
    test_fig: plt.Figure
    test_ax: plt.Axes
    train_canvas: FigureCanvasTkAgg
    test_canvas: FigureCanvasTkAgg
    env: gym.Env

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Reinforcement Learning GUI")
        self.Q_ = None
        self.success_rate = None

        # Left Frame
        self.left_frame = tk.Frame(root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right Frame
        self.right_frame = tk.Frame(root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Radio Buttons for Environment Selection
        self.env_var = tk.StringVar(value="FrozenLake-v1")
        self.create_radio_buttons()

        # Train Header and Inputs
        self.create_train_inputs()

        # Test Header and Inputs
        self.create_test_inputs()

        # Create save button
        save_button = tk.Button(self.left_frame, text="Save Model", command=self.save_model)
        save_button.pack(pady=10)

        # Matplotlib Figures
        self.create_plots()

    def create_radio_buttons(self) -> None:
        env_label = tk.Label(self.left_frame, text="Choose Environment:")
        env_label.pack(anchor=tk.W, padx=10, pady=5)

        environments = [("FrozenLake-v1", "FrozenLake-v1"),
                        ("Taxi-v3", "Taxi-v3"),
                        ("CliffWalking-v0", "CliffWalking-v0")]

        for text, value in environments:
            radio_button = tk.Radiobutton(self.left_frame, text=text, variable=self.env_var, value=value)
            radio_button.pack(anchor=tk.W, padx=20)

    def create_train_inputs(self) -> None:
        train_label = tk.Label(self.left_frame, text="Train", font=("Arial", 14))
        train_label.pack(anchor=tk.W, padx=10, pady=10)

        self.num_episodes_var = tk.IntVar(value=10000)
        self.lr_var = tk.DoubleVar(value=0.1)
        self.gamma_var = tk.DoubleVar(value=0.99)

        self.create_input_field("Number of Episodes:", self.num_episodes_var)
        self.create_input_field("Learning Rate (LR):", self.lr_var)
        self.create_input_field("Gamma:", self.gamma_var)

        train_button = tk.Button(self.left_frame, text="Train", command=self.train)
        train_button.pack(pady=10)

    def create_test_inputs(self) -> None:
        test_label = tk.Label(self.left_frame, text="Test", font=("Arial", 14))
        test_label.pack(anchor=tk.W, padx=10, pady=10)

        self.num_test_episodes_var = tk.IntVar(value=100)

        self.create_input_field("Number of Test Episodes:", self.num_test_episodes_var)

        test_button = tk.Button(self.left_frame, text="Test", command=self.test)
        test_button.pack(pady=10)

    def create_input_field(self, label_text: str, variable: tk.Variable) -> None:
        frame = tk.Frame(self.left_frame)
        frame.pack(anchor=tk.W, padx=20, pady=5)

        label = tk.Label(frame, text=label_text)
        label.pack(side=tk.LEFT)

        entry = tk.Entry(frame, textvariable=variable)
        entry.pack(side=tk.RIGHT)

    def create_plots(self) -> None:
        # Create a frame for the train plot
        train_frame = tk.Frame(self.right_frame)
        train_frame.grid(row=0, column=0, sticky="nsew")

        # Create a frame for the test plot
        test_frame = tk.Frame(self.right_frame)
        test_frame.grid(row=1, column=0, sticky="nsew")

        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        self.train_fig, self.train_ax = plt.subplots()
        self.test_fig, self.test_ax = plt.subplots()

        # Set titles and labels for the train plot
        self.train_ax.set_title("Training Performance")
        self.train_ax.set_xlabel("Episodes")
        self.train_ax.set_ylabel("Reward")

        # Set titles and labels for the test plot
        self.test_ax.set_title("Testing Performance")
        self.test_ax.set_xlabel("Episodes")
        self.test_ax.set_ylabel("Reward")

        self.train_canvas = FigureCanvasTkAgg(self.train_fig, master=train_frame)
        self.train_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.test_canvas = FigureCanvasTkAgg(self.test_fig, master=test_frame)
        self.test_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def reset_q_table(self, env: gym.Env) -> None:
        self.Q_ = np.zeros([env.observation_space.n, env.action_space.n])

    def moving_average(self, data: List[float], window_size: int) -> np.ndarray:
        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

    def train(self) -> None:
        # Placeholder for your training logic
        chosen_env = self.env_var.get()
        num_episodes = self.num_episodes_var.get()
        lr = self.lr_var.get()
        gamma = self.gamma_var.get()
        print(f"Training {chosen_env} for {num_episodes} episodes with LR={lr} and gamma={gamma}")
        print(f"Initializing environment")
        self.env = gym.make(chosen_env, render_mode='ansi')
        success_rate = []
        rewards = []
        steps = []
        self.Q_ = np.zeros((self.env.observation_space.n, self.env.action_space.n))  # Initialize Q-table

        # Run the Q-learning algorithm
        for episode in range(num_episodes):
            # Reset the environment and get the initial state
            state, _ = self.env.reset()
            step = 0
            done = False
            truncated = False
            total_reward = 0

            # Decaying epsilon-greedy strategy
            epsilon = max(0.01, 1.0 - episode / (0.9 * num_episodes))

            while not done and not truncated:
                # Choose an action using epsilon-greedy policy
                if np.random.uniform(0, 1) < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Q_[state, :])

                # Take the action and observe the next state and reward
                next_state, reward, done, truncated, _ = self.env.step(action)

                # Update the Q_-value for the current state-action pair
                self.Q_[state, action] = self.Q_[state, action] + lr * (
                        reward + gamma * np.max(self.Q_[next_state, :]) - self.Q_[state, action])

                total_reward += reward
                state = next_state
                step += 1

            if truncated:
                total_reward -= 10.0
            rewards.append(total_reward)
            steps.append(step)

            # Calculate and append the success rate every 5000 episodes
            if (episode + 1) % 5000 == 0:
                success_rate.append(np.mean(rewards[-5000:]))

        # Calculate the overall success rate
        success_rate = np.mean([1 if reward > 0 else 0 for reward in rewards])
        print(f"Overall success rate: {success_rate}")
        self.print_train_canvas(rewards)

    def print_train_canvas(self, values: List[float]) -> None:
        # Update the train plot here
        self.train_ax.clear()
        window_size = 100  # Adjust window_size as needed
        if len(values) >= window_size:
            smoothed_values = self.moving_average(values, window_size)
            self.train_ax.plot(smoothed_values)
        else:
            self.train_ax.plot(values)
        # Set titles and labels for the train plot
        self.train_ax.set_title("Training Performance")
        self.train_ax.set_xlabel("Episodes")
        self.train_ax.set_ylabel("Reward")
        self.train_canvas.draw()

    def test(self) -> None:
        print(f"Testing {self.env_var.get()} for {self.num_test_episodes_var.get()} episodes")
        total_reward = 0
        success_count = 0
        total_steps = 0
        test_rewards = []
        num_test_episodes = self.num_test_episodes_var.get()

        for _ in range(num_test_episodes):
            state, _ = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_steps = 0

            while not done and not truncated:
                action = np.argmax(self.Q_[state, :])
                state, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward
                episode_steps += 1

            total_reward += episode_reward
            total_steps += episode_steps
            test_rewards.append(episode_reward)
            if episode_reward > 0:
                success_count += 1

        print(f"Finished testing")
        self.print_test_canvas(test_rewards)

    def print_test_canvas(self, values: List[int]) -> None:
        # Update the test plot here
        self.test_ax.clear()
        window_size = 10  # Adjust window_size as needed
        if len(values) >= window_size:
            smoothed_values = self.moving_average(values, window_size)
            self.test_ax.plot(smoothed_values)
        else:
            self.test_ax.plot(values)
        # Set titles and labels for the test plot
        self.test_ax.set_title("Testing Performance")
        self.test_ax.set_xlabel("Episodes")
        self.test_ax.set_ylabel("Reward")
        self.test_canvas.draw()

    def save_model(self) -> None:
        chosen_env = self.env_var.get()
        num_episodes = self.num_episodes_var.get()
        lr = self.lr_var.get()
        gamma = self.gamma_var.get()

        # Create folder name
        folder_name = f"{chosen_env}-ne-{num_episodes}-lr-{lr}-gamma-{gamma}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Save the Q-table
        np.save(os.path.join(folder_name, "Q_table.npy"), self.Q_)

        # Save the training plot
        self.train_fig.savefig(os.path.join(folder_name, "training_performance.png"))

        # Save the testing plot
        self.test_fig.savefig(os.path.join(folder_name, "testing_performance.png"))

        # Save the hyperparameters
        with open(os.path.join(folder_name, "hyperparams.txt"), "w") as f:
            f.write(f"NUM_EPISODES={num_episodes}\n")
            f.write(f"LEARNING_RATE={lr}\n")
            f.write(f"GAMMA={gamma}\n")

        print(f"Model and data saved in folder: {folder_name}")


if __name__ == "__main__":
    root = tk.Tk()
    app = RLGuiApp(root)
    root.mainloop()

