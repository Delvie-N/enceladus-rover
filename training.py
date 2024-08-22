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
import os
import time

class EnceladusNetwork(nn.Module):
    """ This class contains the setup of the Neural Network required for employing DQN for exploring Enceladus """

    def __init__(self, observation_space_dimensions, action_space_dimensions) -> None:
        super().__init__()

        hidden_layer1 = 32
        hidden_layer2 = 16

        padding = 0
        dilation = 1
        kernel_size = 3
        stride = 1
        conv_input_size = EnceladusEnvironment().grid_height
        conv_output_size_1 = np.int64(((conv_input_size + 2*padding - dilation*(kernel_size-1) - 1)/stride) + 1)
        conv_output_size_2 = np.int64(((conv_output_size_1 + 2*padding - dilation*(kernel_size-1) - 1)/stride) + 1)
        conv_output_size_3 = np.int64(((conv_output_size_2 + 2*padding - dilation*(kernel_size-1) - 1)/stride) + 1)

        self.shared_network = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size), # originally nn.Conv2d(1, 1, kernel_size),
            nn.Tanh(),
            nn.Conv2d(6, 12, kernel_size), # originally nn.Conv2d(1, 1, kernel_size),
            nn.Tanh(),
            nn.Conv2d(12, 12, kernel_size), # originally nn.Conv2d(1, 1, kernel_size),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(conv_output_size_3**2, hidden_layer1),
            nn.Tanh(), # originally (and alternated with) nn.Sigmoid()
            nn.Linear(hidden_layer1, hidden_layer2),
            nn.Tanh(), # originally (and alternated with) nn.Sigmoid()
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
    """ This class contains the implementation of the NN from EnceladusNetwork by taking the observation and action space dimensions to put into the NN and calculate the probabilities for each action and returning the optimal action according to the epsilon-greedy policy """

    def __init__(self, observation_space_dimensions, action_space_dimensions):
        self.learning_rate = 1e-4 # originally 1e-3
        self.gamma = 1 - 1e-2 # originally 1 - 1e-2
        self.epsilon = 1e-5 # originally 1e-6

        self.probabilities = []
        self.rewards = []

        self.network = EnceladusNetwork(observation_space_dimensions, action_space_dimensions)
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.learning_rate) # originally torch.optim.SGD(self.network.parameters(), lr=self.learning_rate)

    def sample_action(self, observations):
        """ This function takes the observation space (grid world and rover position) to determine the rover's next action """

        state = [observations['position_x'], observations['position_y']]
        state.extend(list(np.ndarray.flatten(observations['world'])))

        state = torch.tensor(np.array([state]))
        tensor_image_input = observations['world']
        tensor_image_input = tensor_image_input.reshape((tensor_image_input.shape[0], tensor_image_input.shape[1], 1))
        tensor_image = torch.from_numpy(tensor_image_input)

        tensor_image = torch.permute(tensor_image, (2, 0, 1))

        action_means, action_std = self.network(tensor_image)

        distribution = Normal(action_means[0] + self.epsilon, action_std[0] + self.epsilon)
        action_tensor = distribution.sample()

        probability = distribution.log_prob(action_tensor)

        action_array = action_tensor.numpy()

        self.probabilities.append(probability)
         
        action = np.where(action_array==max(action_array))[0][0]

        if (random.random() < self.epsilon):                        
            action = np.random.randint(0,7)

        return action
    
    def update(self):
        """ This function is used to update the agent/NN after finishing an episode set """

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

start_time_print = time.ctime()
start_time = time.time()

print('\nModel started running at:', start_time_print, '\n') 

env = EnceladusEnvironment()
print('Hello Icy World')
print('Sensitivity analysis for epsilon = 1e-5 instead of 1e-6 ')
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)

total_episode_amount = int(400000)
total_seed_amount = int(1)

observations = env.get_observations()
state = [observations['position_x'], observations['position_y']]
state.extend(list(np.ndarray.flatten(observations['world'])))
observation_space_dimensions = len(state)

action_space_dimensions = 8

rewards_over_seeds = []

weight_path = f'weights/{total_episode_amount}-steps-{total_seed_amount}-seeds.pth'
weight_file_version = 1

while os.path.isfile(weight_path) is True:
    if weight_file_version == 1:
        weight_path = weight_path.split('.')[0] + f'-{weight_file_version}.pth'
    else:
        weight_path = weight_path.replace(f'-{weight_file_version-1}.pth', f'-{weight_file_version}.pth')
    weight_file_version += 1

agent = RoverTraining(observation_space_dimensions, action_space_dimensions)
model = agent.network

seed_number = 0

for seed in np.random.randint(0, 500, size=total_seed_amount, dtype=int): # alternated with np.arange(1, total_seed_amount+1):
    high_score = -1000
    seed = int(seed)
    seed_number += 1

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    rewards_over_episodes = []
    mission_success_over_episodes = []

    for episode in range(total_episode_amount): 
        observations, information = wrapped_env.reset(seed=seed)
        done = False
        mission_success = False
        score = 0
        while not done:
            action = agent.sample_action(observations)
            observations, reward, terminated, truncated, info = wrapped_env.step(action)

            unpassed_samplearea = np.count_nonzero(observations['world'] == 3)
            unpassed_endpoint = np.count_nonzero(observations['world'] == 4)
            if unpassed_samplearea < 8 and unpassed_endpoint < 1:
                mission_success = True
            
            agent.rewards.append(reward)
            score += reward

            done = terminated

        if score > high_score:
           high_score = score

           figure_highscore, axes_highscore = plt.subplots(figsize=(6, 6))

           axes_highscore.imshow(env.surface_grid.transpose(), cmap=env.cmap)
           axes_highscore.scatter(env.start_x, env.start_y, color='springgreen', label='Start', marker='s')
           axes_highscore.scatter(env.end_x, env.end_y, color='red', label='End', marker='s')

           axes_highscore.grid(False)
           figure_highscore.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

           file_path_highscore = 'visuals/highscores/rover_training_highscore.jpg'
           file_version_highscore = 1

           while os.path.isfile(file_path_highscore) is True:
               if file_version_highscore == 1:
                   file_path_highscore = file_path_highscore.split('.')[0] + f'-{file_version_highscore}.jpg'
               else:
                   file_path_highscore = file_path_highscore.replace(f'-{file_version_highscore-1}.jpg', f'-{file_version_highscore}.jpg')
               file_version_highscore += 1

           plt.savefig(file_path_highscore, dpi=150)
           plt.close()
    
        rewards_over_episodes.append(wrapped_env.return_queue[-1])
        mission_success_over_episodes.append(mission_success)
        agent.update()

        if episode % 1 == 0:
            average_reward = int(np.mean(wrapped_env.return_queue))
            average_mission_success_over_episodes = np.mean(mission_success_over_episodes)
            print(f'seed {seed_number}: {seed} so {seed_number/total_seed_amount:.1%} | {episode} so {episode/total_episode_amount:>4.1%} | {average_reward} | average mission success: {average_mission_success_over_episodes:.1%} \t', end='\r')

    rewards_over_seeds.append(rewards_over_episodes)

torch.save(model.state_dict(), weight_path)

rewards_to_plot = []
iteration = []
for rewards in rewards_over_seeds:
    for reward in rewards:
        rewards_to_plot.append(reward[0])

running_mean_size = int((total_episode_amount*total_seed_amount)/100)
half_running_mean_size = int(running_mean_size/2)
min_running_mean_index = half_running_mean_size
max_running_mean_index = len(rewards_to_plot) - (half_running_mean_size)

running_means_to_plot = []
running_means_index_to_plot = list(np.arange(min_running_mean_index, max_running_mean_index, 1))

for running_mean_index in range(min_running_mean_index, max_running_mean_index):
    running_mean_sum = 0
    for i in range(running_mean_index-(half_running_mean_size), running_mean_index+(half_running_mean_size)):
        running_mean_sum += rewards_to_plot[i]

    running_mean = running_mean_sum/(running_mean_size+1)
    running_means_to_plot.append(running_mean)

df1 = pd.DataFrame([rewards_to_plot]).melt()
df1.rename(columns={'variable': 'episodes', 'value': 'reward'}, inplace=True)
sns.set(style='darkgrid', context='talk', palette='rainbow')
sns.scatterplot(x='episodes', y='reward', data=df1).set(
    title='RoverTraining for EnceladusNetwork')
plt.plot(running_means_index_to_plot, running_means_to_plot, color='red')

result_file_path = 'training_results/tanh/result_tanh_32_16.png' # originally 'training_results/sigmoid/result_sigmoid_20_10.png'
result_file_version = 1

while os.path.isfile(result_file_path) is True:
    if result_file_version == 1:
        result_file_path = result_file_path.split('.')[0] + f'-{result_file_version}.png'
    else:
        result_file_path = result_file_path.replace(f'-{result_file_version-1}.png', f'-{result_file_version}.png')
    result_file_version += 1

plt.savefig(result_file_path)

end_time_print = time.ctime()
end_time = time.time()
run_time = end_time - start_time

print('\n \nModel stopped training at:', end_time_print)
print('Total runtime equals:', round(run_time, 2), 'seconds\n')

plt.show()

env.reset(seed=seed)
figure, axes = plt.subplots(figsize=(6, 6))

frames = []
fps = 10

n_steps = 400
observations = env.get_observations()
agent.epsilon = 0

for step in range(n_steps):
    action = agent.sample_action(observations)     

    print('Step:', step+1)
    observations, reward, done, _, _ = env.step(action)

    unpassed_samplearea = np.count_nonzero(observations['world'] == 3)
    unpassed_endpoint = np.count_nonzero(observations['world'] == 4)
    if unpassed_samplearea < 8 and unpassed_endpoint < 1:
        mission_success = True

    print('Reward:', reward)
    print('Mission success:', mission_success)
    print('Done?:', done, '\n')

    new_frame = [axes.imshow(env.surface_grid.transpose(), cmap=env.cmap),
                axes.scatter(env.start_x, env.start_y, color='springgreen', label='Start', marker='s'),
                axes.scatter(env.end_x, env.end_y, color='red', label='End', marker='s')]
        
    frames.append(new_frame)
    if done:
        break

axes.grid(False)
figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
rover_animation = animation.ArtistAnimation(figure, frames, interval=int(1000/fps), blit=True, repeat_delay=1000)

file_path = 'visuals/trainings/rover_animation_training.gif'
file_version = 1

while os.path.isfile(file_path) is True:
    if file_version == 1:
        file_path = file_path.split('.')[0] + f'-{file_version}.gif'
    else:
        file_path = file_path.replace(f'-{file_version-1}.gif', f'-{file_version}.gif')
    file_version += 1

rover_animation.save(file_path, dpi=150)
