# Maintient d'un taux minimal d'exploration (`epsilon_min`) même après de nombreuses itérations d'entraînement

Maintenir un taux minimal d'exploration (`epsilon_min`) est crucial pour plusieurs raisons :

1. **Éviter les boucles infinies** : Même si l'agent a trouvé une stratégie optimale, l'environnement peut changer légèrement ou des états peuvent être explorés de manière inattendue. Une exploration minimale permet à l'agent de découvrir ces changements potentiels.
2. **Découverte de meilleures stratégies** : L'environnement peut contenir des stratégies plus efficaces que celles actuellement exploitées. Une faible probabilité d'exploration permet de découvrir et d'adopter ces stratégies.
3. **Prévention du surapprentissage** : Si l'agent exploite toujours les mêmes actions, il pourrait surapprendre une stratégie qui n'est pas réellement optimale à long terme. Un peu d'exploration aide à généraliser les apprentissages.

Le code ci-dessous illustre l'utilisation de l'algorithme Q-Learning avec le maintien d'un taux minimal d'exploration `epsilon_min`.

```python
import numpy as np
import gym
import random
from IPython.display import clear_output
from time import sleep

# Initialize the Taxi-v3 environment
env = gym.make("Taxi-v3").env

# Initialize the Q-table with zeros
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Set the hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.6  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1
epsilon_decay = 0.995

# For logging metrics
all_epochs = []
all_penalties = []

# Training loop
for i in range(1, 100001):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done, _ = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        # Update Q-value for the current state-action pair
        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
    
    # Decay the exploration rate
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    # Log progress every 100 episodes
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")

# Evaluate the trained agent
total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, _ = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
```

Maintenir une exploration minimale aide à garantir que l'agent ne se contente pas de la première solution trouvée, mais continue à chercher des solutions potentiellement meilleures.

Cela aide également à éviter que l'agent ne soit trop rigide dans des environnements dynamiques.
