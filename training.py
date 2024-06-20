from map import EnceladusEnvironment
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import gymnasium as gym
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from time import sleep
import os

class EnceladusNetwork(nn.Module):
    """
    asdf
    """

    def __init__(self, observation_space_dimensions, action_space_dimensions) -> None:
        super().__init__()

        hidden_layer1 = 20
        hidden_layer2 = 10

        self.shared_network = nn.Sequential(
            nn.Linear(observation_space_dimensions, hidden_layer1),
            nn.Sigmoid(),
            nn.Linear(hidden_layer1, hidden_layer2),
            nn.Sigmoid(),
        )

        self.policy_mean_network = nn.Sequential(
            nn.Linear(hidden_layer2, action_space_dimensions)
        )

        self.policy_std_network = nn.Sequential(
            nn.Linear(hidden_layer2, action_space_dimensions)
        )

    def forward(self, x: torch.Tensor):
        shared_features = self.shared_network(x.float())

        action_means = self.policy_mean_network(shared_features)
        action_std = torch.log(1+torch.exp(self.policy_std_network(shared_features)))

        return action_means, action_std

class RoverTraining():
    """
    asdf
    """

    def __init__(self, observation_space_dimensions, action_space_dimensions):
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon = 1e-6

        self.probabilities = []
        self.rewards = []

        self.network = EnceladusNetwork(observation_space_dimensions, action_space_dimensions)
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.learning_rate)

    def sample_action(self, state):
        state = torch.tensor(np.array([state]))
        action_means, action_std = self.network(state)

        distribution = Normal(action_means[0] + self.epsilon, action_std[0] + self.epsilon)
        action = distribution.sample()
        probability = distribution.log_prob(action)

        action = action.numpy()

        self.probabilities.append(probability)

        return action
    
    def update(self):
        running_gamma = 0
        gammas = []

        for R in self.rewards[::-1]:
            running_gamma = R + self.gamma * running_gamma
            gammas.insert(0, running_gamma)

        deltas = torch.tensor(gammas)

        loss = 0

        for log_prob, delta in zip(self.probabilities, deltas):
            loss += log_prob.mean() * delta * (-1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.probabilities = []
        self.rewards = []

env = EnceladusEnvironment()
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)

total_episode_amount = int(1000)
total_seed_amount = int(1)

observations = env.get_observations()
state = [observations['position_x'], observations['position_y']]
state.extend(list(np.ndarray.flatten(observations['world'])))
observation_space_dimensions = len(state)

action_space_dimensions = 8

rewards_over_seeds = []

weight_path = f'weights/{total_episode_amount}-steps-{total_seed_amount}-seeds.pth'
agent = RoverTraining(observation_space_dimensions, action_space_dimensions)
model = agent.network

seed_number = 0

for seed in [100]:#np.random.randint(0, 500, size=total_seed_amount, dtype=int):
    seed = int(seed)
    seed_number += 1

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    rewards_over_episodes = []

    for episode in range(total_episode_amount):
        observations, information = wrapped_env.reset(seed=seed)
        done = False
        while not done:
            state = [observations['position_x'], observations['position_y']]
            state.extend(list(np.ndarray.flatten(observations['world'])))
            action_array = agent.sample_action(state)
            
            action = np.where(action_array==max(action_array))[0][0]

            observations, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)

            done = terminated
        
        rewards_over_episodes.append(wrapped_env.return_queue[-1])
        agent.update()

        if episode % 1 == 0:
            average_reward = int(np.mean(wrapped_env.return_queue))
            print(f'seed {seed_number:<6}: {seed:>3} so {seed_number/total_seed_amount:>4.1%}\t | \t {episode :>4} so {episode/total_episode_amount:>4.1%} \t | \t {average_reward :<20}', end='\r')

    rewards_over_seeds.append(rewards_over_episodes)
    
torch.save(model.state_dict(), weight_path)

rewards_to_plot = []
iteration = []
for rewards in rewards_over_seeds:
    for reward in rewards:
        rewards_to_plot.append(reward[0])
df1 = pd.DataFrame([rewards_to_plot]).melt()
df1.rename(columns={'variable': 'episodes', 'value': 'reward'}, inplace=True)
sns.set(style='darkgrid', context='talk', palette='rainbow')
sns.scatterplot(x='episodes', y='reward', data=df1).set(
    title='RoverTraining for EnceladusNetwork')

result_file_path = 'training_results/result_sigmoid_20_10.png'
result_file_version = 1

while os.path.isfile(result_file_path) is True:
    if result_file_version == 1:
        result_file_path = result_file_path.split('.')[0] + f'-{result_file_version}.png'
    else:
        result_file_path = result_file_path.replace(f'-{result_file_version-1}.png', f'-{result_file_version}.png')
    result_file_version += 1

plt.savefig(result_file_path)
plt.show()
env = EnceladusEnvironment()

figure, axes = plt.subplots(figsize=(6, 6))

frames = []
fps = 10

n_steps = 400
observations = env.get_observations()

for step in range(n_steps):
    state = [observations['position_x'], observations['position_y']]
    state.extend(list(np.ndarray.flatten(observations['world'])))
    action_array = agent.sample_action(state)     
    action = np.where(action_array==max(action_array))[0][0]

    print('Step:', step+1)
    observations, reward, done, _, _ = env.step(action)
    print('Reward:', reward)
    print('Done?:', done, '\n')

    new_frame = [axes.imshow(env.surface_grid.transpose(), cmap=env.cmap),
                axes.scatter(env.start_x, env.start_y, color='springgreen', label='Start', marker='s'),
                axes.scatter(env.end_x, env.end_y, color='red', label='End', marker='s')]
        
    frames.append(new_frame)

    if done:
        print('DONEEEEEEEEEEEEEE')
        break

axes.grid(False)
figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
rover_animation = animation.ArtistAnimation(figure, frames, interval=int(1000/fps), blit=True, repeat_delay=1000)

file_path = 'animations/rover_animation_training.gif'
file_version = 1

while os.path.isfile(file_path) is True:
    if file_version == 1:
        file_path = file_path.split('.')[0] + f'-{file_version}.gif'
    else:
        file_path = file_path.replace(f'-{file_version-1}.gif', f'-{file_version}.gif')
    file_version += 1

rover_animation.save(file_path, dpi=150)