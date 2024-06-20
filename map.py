import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import gymnasium as gym

class EnceladusEnvironment(gym.Env):
	"""
	asdf
	"""
	TYPE = {'empty': 0,
			'end': 1,
			'passed': 2,
			'rover': 3,
			'ridge': 4
			}

	def __init__(self) -> None:
		print('Hello Icy World')
		self.generate_grid()

		#Observation space:
		self.observation_space = gym.spaces.Dict(spaces={
			'world': gym.spaces.Box(low=0, high=3, shape=(self.grid_width, self.grid_height), dtype=np.uint8),
			'position_x': gym.spaces.Box(low=0, high=(self.grid_width-1), shape=(1,), dtype=np.int32),
			'position_y': gym.spaces.Box(low=0, high=(self.grid_height-1), shape=(1,), dtype=np.int32)
		})

		#Action space
		self.action_space = gym.spaces.Discrete(8)

	def get_observations(self):
		return {'world': self.surface_grid,
				'position_x': self.rover_x,
				'position_y': self.rover_y
				}

	def generate_grid(self):
		self.grid_width = 40
		self.grid_height = 40
		self.surface_grid = np.zeros((self.grid_width, self.grid_height))

		self.fixed_point_distance = 5

		self.start_x = self.fixed_point_distance
		self.start_y = self.grid_height-self.fixed_point_distance
		self.end_x = self.grid_width-self.fixed_point_distance
		self.end_y = self.fixed_point_distance

		self.rover_x = self.start_x
		self.rover_y = self.start_y

		self.ridge_amount = np.random.randint(6,12)

		for ridge_i in range(self.ridge_amount):
			ridge_size_max = np.random.randint(6,18)
			
			#ridge_start_location_x = np.random.randint(0, self.grid_width)
			ridge_start_location_x = np.random.randint(self.start_x, self.end_x)
			#ridge_start_location_y = np.random.randint(0, self.grid_height)
			ridge_start_location_y = np.random.randint(self.end_y, self.start_y)

			self.surface_grid[ridge_start_location_x, ridge_start_location_y] = self.TYPE['ridge']

			ridge_location_x = ridge_start_location_x
			ridge_location_y = ridge_start_location_y

			for i in range(ridge_size_max-1):
				ridge_location_x += np.random.choice([-1, 0, 1])
				ridge_location_y += np.random.choice([-1, 0, 1])
				self.surface_grid[ridge_location_x, ridge_location_y] = self.TYPE['ridge']

		for boundary_point_x in [-1, 0, 1]:
			clearup_start_x = self.start_x + boundary_point_x
			clearup_end_x = self.end_x + boundary_point_x
			for boundary_point_y in [-1, 0, 1]:
				clearup_start_y = self.start_y + boundary_point_y
				clearup_end_y = self.end_y + boundary_point_y
				self.surface_grid[clearup_start_x, clearup_start_y] = self.TYPE['empty']
				self.surface_grid[clearup_end_x, clearup_end_y] = self.TYPE['empty']

		self.surface_grid[self.rover_x, self.rover_y] = self.TYPE['rover']
		self.surface_grid[self.end_x, self.end_y] = self.TYPE['end']

		self.cmap = colors.ListedColormap(['lightskyblue', 'red', 'lightgrey', 'black', 'steelblue'])
		# plt.figure(figsize=(6, 6))
		# plt.title('Exploring Enceladus')
		# plt.imshow(self.surface_grid.transpose(), cmap=self.cmap)
		# plt.scatter(self.start_x, self.start_y, color='springgreen', label='Start', marker='s')
		# plt.scatter(self.end_x, self.end_y, color='red', label='End', marker='s')
		# plt.legend()
		# plt.show()
	
	def step(self, action):
		done = False

		self.steps = np.array([[0, 1, 1, 1, 0, -1, -1, -1],
							   [1, 1, 0, -1, -1, -1, 0, 1]])
		
		self.surface_grid[self.rover_x, self.rover_y] = self.TYPE['passed']

		self.initial_difference_x = np.abs(self.end_x - self.rover_x)
		self.initial_difference_y = np.abs(self.end_y - self.rover_y)

		self.rover_x += self.steps[0, action]
		self.rover_y += self.steps[1, action]

		self.new_difference_x = np.abs(self.end_x - self.rover_x)
		self.new_difference_y = np.abs(self.end_y - self.rover_y)

		self.reward_x = 0
		self.reward_y = 0

		if self.grid_width > self.rover_x >= 0 and self.grid_height > self.rover_y >=0:
			if self.surface_grid[self.rover_x, self.rover_y] == self.TYPE['ridge']:
				self.reward = -100
				print("CRASHED INTO RIDGE")
				done = True

			self.surface_grid[self.rover_x, self.rover_y] = self.TYPE['rover']

			if self.initial_difference_x > self.new_difference_x:
				self.reward_x = 1
			elif self.initial_difference_x < self.new_difference_x:
				self.reward_x = -2
			if self.rover_x >= self.grid_width-1 or self.rover_x <= 1:
				self.reward_x = -2

			if self.initial_difference_y > self.new_difference_y:
				self.reward_y = 1
			elif self.initial_difference_y < self.new_difference_y:
				self.reward_y = -2			
			if self.rover_y >= self.grid_height-1 or self.rover_y <= 1:
				self.reward_y = -2

			self.reward = self.reward_x + self.reward_y

			if self.rover_x == self.end_x and self.rover_y == self.end_y:
				self.reward = 200
				print("MADE IT TO END POINT")
				done = True

		else:
			self.reward = -100
			done = True

		# plt.figure(figsize=(6, 6))
		# plt.title('Exploring Enceladus')
		# plt.imshow(self.surface_grid.transpose(), cmap=self.cmap)
		# plt.scatter(self.start_x, self.start_y, color='springgreen', label='Start', marker='s')
		# plt.scatter(self.end_x, self.end_y, color='red', label='End', marker='s')
		# plt.legend()
		# plt.show()

		observations = self.get_observations()

		return observations, self.reward, done, {}, {}
	
	def reset(self, seed=None, options = None):
		if seed:
			np.random.seed(seed)
		self.surface_grid = np.zeros((self.grid_width, self.grid_height))
		
		self.start_x = self.fixed_point_distance
		self.start_y = self.grid_height-self.fixed_point_distance
		self.rover_x = self.start_x
		self.rover_y = self.start_y

		self.surface_grid[self.rover_x, self.rover_y] = self.TYPE['rover']
		self.surface_grid[self.end_x, self.end_y] = self.TYPE['end']
		return self.get_observations(), {}
	
if __name__ == "__main__":
	enceladus_environment = EnceladusEnvironment()
	print(enceladus_environment.step(4)[1])
	print(enceladus_environment.step(5)[1])
	print(enceladus_environment.step(3)[1])
	print(enceladus_environment.step(3)[1])
	print(enceladus_environment.step(3)[1])
	print(enceladus_environment.step(4)[1])
	print(enceladus_environment.step(2)[1])
	print(enceladus_environment.step(3)[1])