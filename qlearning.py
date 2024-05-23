import tkinter as tk
import numpy as np
import gymnasium as gym

from base import BaseRLApp


class QLearningApp(BaseRLApp):
    def train(self) -> None:
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


if __name__ == "__main__":
    root = tk.Tk()
    app = QLearningApp(root)
    root.mainloop()
