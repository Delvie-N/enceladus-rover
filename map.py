import numpy as np
from matplotlib import colors
import gymnasium as gym

class EnceladusEnvironment(gym.Env):
	""" This class contains all required elements for generating the Enceladus Environment grid world,
	including the definition of the TYPEs, Observation and Action spaces """
	
	# Commented lines below are the original values associated with the seven different surface TYPEs:
	#
	# TYPE = {'empty': 0,
	# 		'end': 1,
	# 		'passed': 2,
	# 		'rover': 3,
	# 		'ridge': 4,
	# 		'plume': 5,
	# 		'plumesampling': 6
	# 		}
	
	TYPE = {'empty': 2,
			'end': 4,
			'passed': 0,
			'rover': 1,
			'ridge': -2,
			'plume': -1,
			'plumesampling': 3
			}

	def __init__(self) -> None:
		self.generate_grid()

		self.observation_space = gym.spaces.Dict(spaces={
			'world': gym.spaces.Box(low=0, high=3, shape=(self.grid_width, self.grid_height), dtype=np.uint8),
			'position_x': gym.spaces.Box(low=0, high=(self.grid_width-1), shape=(1,), dtype=np.int32),
			'position_y': gym.spaces.Box(low=0, high=(self.grid_height-1), shape=(1,), dtype=np.int32)
		})

		self.action_space = gym.spaces.Discrete(8)

	def get_observations(self):
		""" This function retrieves the current environment ('world') and position of the rover ('position_x' 
		and 'position_y') """

		return {'world': self.surface_grid,
				'position_x': self.rover_x,
				'position_y': self.rover_y
				}

	def generate_grid(self):
		""" This function generates the actual surface grid world and adds start and end locations, icy ridges
		and a plume vent """

		self.grid_width = 20 # originally 40
		self.grid_height = 20 # originally 40
		self.surface_grid = np.zeros((self.grid_width, self.grid_height))
		self.surface_grid[:,:] = self.TYPE['empty']

		self.fixed_point_distance = 3 # originally 5

		self.start_x = self.fixed_point_distance
		self.start_y = self.grid_height-(self.fixed_point_distance+1)
		self.end_x = self.grid_width-(self.fixed_point_distance+1)
		self.end_y = self.fixed_point_distance

		self.ridge_amount = np.random.randint(6,8) # originally np.random.randint(6,12)

		for ridge_i in range(self.ridge_amount):
			ridge_size_max = np.random.randint(6,12) # originally np.random.randint(6,18)
			
			ridge_start_location_x = np.random.randint(self.start_x, self.end_x) # originally np.random.randint(0, self.grid_width)
			ridge_start_location_y = np.random.randint(self.end_y, self.start_y) # originally np.random.randint(0, self.grid_height)

			self.surface_grid[ridge_start_location_x, ridge_start_location_y] = self.TYPE['ridge']

			ridge_location_x = ridge_start_location_x
			ridge_location_y = ridge_start_location_y

			for i in range(ridge_size_max-1):
				ridge_location_x += np.random.choice([-1, 0, 1])
				ridge_location_y += np.random.choice([-1, 0, 1])
				if (0 < ridge_location_x < self.grid_width-1) and (0 < ridge_location_y < self.grid_height-1):
					self.surface_grid[ridge_location_x, ridge_location_y] = self.TYPE['ridge']

		self.plume_location_x = np.random.randint(self.start_x + 5, self.end_x - 5)
		self.plume_location_y = np.random.randint(self.end_y + 5, self.start_y - 5)

		for boundary_point_x in [-3, -2, -1, 0, 1, 2, 3]:
			clearup_plume_x = self.plume_location_x + boundary_point_x
			for boundary_point_y in [-3, -2, -1, 0, 1, 2, 3]:
				clearup_plume_y = self.plume_location_y + boundary_point_y
				self.surface_grid[clearup_plume_x, clearup_plume_y] = self.TYPE['empty']

		self.plume_samplearea = []

		for boundary_point_x in [-1, 0, 1]:
			plume_samplearea_x = self.plume_location_x + boundary_point_x
			for boundary_point_y in [-1, 0, 1]:
				plume_samplearea_y = self.plume_location_y + boundary_point_y
				self.surface_grid[plume_samplearea_x, plume_samplearea_y] = self.TYPE['plumesampling']
				self.plume_samplearea.append([plume_samplearea_x, plume_samplearea_y])

		self.surface_grid[self.plume_location_x, self.plume_location_y] = self.TYPE['plume']

		for boundary_point_x in [-2, -1, 0, 1, 2]:
			clearup_start_x = self.start_x + boundary_point_x
			clearup_end_x = self.end_x + boundary_point_x
			for boundary_point_y in [-2, -1, 0, 1, 2]:
				clearup_start_y = self.start_y + boundary_point_y
				clearup_end_y = self.end_y + boundary_point_y
				self.surface_grid[clearup_start_x, clearup_start_y] = self.TYPE['empty']
				self.surface_grid[clearup_end_x, clearup_end_y] = self.TYPE['empty']

		self.rover_x = self.start_x
		self.rover_y = self.start_y

		self.surface_grid[self.rover_x, self.rover_y] = self.TYPE['rover']
		self.surface_grid[self.end_x, self.end_y] = self.TYPE['end']

		self.cmap = colors.ListedColormap(['steelblue', 'aquamarine', 'lightgrey', 'black','lightskyblue', 'mediumaquamarine', 'red']) # originally colors.ListedColormap(['lightskyblue', 'red', 'lightgrey', 'black', 'steelblue', 'aquamarine', 'mediumaquamarine'])
	
	def step(self, action):
		done = False
		self.plume_sampled = False

		self.steps = np.array([[0, 1, 1, 1, 0, -1, -1, -1],
							   [1, 1, 0, -1, -1, -1, 0, 1]])
		
		if self.grid_width-1 >= self.rover_x >= 0 and self.grid_height-1 >= self.rover_y >=0 and [self.rover_x, self.rover_y] != [self.end_x, self.end_y]:
			self.surface_grid[self.rover_x, self.rover_y] = self.TYPE['passed']

		for i in range(len(self.plume_samplearea)):
			x_check = self.plume_samplearea[i][0]
			y_check = self.plume_samplearea[i][1]
			if self.surface_grid[x_check, y_check] == self.TYPE['passed']:
				self.plume_sampled = True
		
		if self.plume_sampled == False:
			self.initial_difference_x = np.abs(self.plume_location_x - self.rover_x)
			self.initial_difference_y = np.abs(self.plume_location_y - self.rover_y)

		if self.plume_sampled == True:
			self.initial_difference_x = np.abs(self.end_x - self.rover_x)
			self.initial_difference_y = np.abs(self.end_y - self.rover_y)

		self.rover_x += self.steps[0, action]
		self.rover_y += self.steps[1, action]

		if self.plume_sampled == False:
			self.new_difference_x = np.abs(self.plume_location_x - self.rover_x)
			self.new_difference_y = np.abs(self.plume_location_y - self.rover_y)

		if self.plume_sampled == True:
			self.new_difference_x = np.abs(self.end_x - self.rover_x)
			self.new_difference_y = np.abs(self.end_y - self.rover_y)

		self.reward = 0
		self.reward_x = 0
		self.reward_y = 0
		
		if self.grid_width-1 >= self.rover_x >= 0 and self.grid_height-1 >= self.rover_y >= 0:
			if self.initial_difference_x > self.new_difference_x:
				self.reward_x = 1
			if self.initial_difference_x < self.new_difference_x:
				self.reward_x = -1
			if self.initial_difference_y > self.new_difference_y:
				self.reward_y = 1
			if self.initial_difference_y < self.new_difference_y:
				self.reward_y = -1
			
			# Commented lines below are part of an old rewards strategy where a gradient of increasingly positive rewards was put around the plume vent:

			# if self.plume_sampled == False and (self.plume_location_x-4 <= self.rover_x <= self.plume_location_x+4) and (self.plume_location_y-4 <= self.rover_y <= self.plume_location_y+4):
			# 	if self.initial_difference_x >= self.new_difference_x:
			# 		if self.rover_x == self.plume_location_x+4 or self.rover_x == self.plume_location_x-4:
			# 			self.reward_x = 2 #5
			# 		if self.rover_x == self.plume_location_x+3 or self.rover_x == self.plume_location_x-3:
			# 			self.reward_x = 3 #10
			# 		if self.rover_x == self.plume_location_x+2 or self.rover_x == self.plume_location_x-2:
			# 			self.reward_x = 4 #15
			# 	if self.initial_difference_y >= self.new_difference_y:
			# 		if self.rover_y == self.plume_location_y+4 or self.rover_y == self.plume_location_y-4:
			# 			self.reward_y = 2 #5
			# 		if self.rover_y == self.plume_location_y+3 or self.rover_y == self.plume_location_y-3:
			# 			self.reward_y = 3 #10
			# 		if self.rover_y == self.plume_location_y+2 or self.rover_y == self.plume_location_y-2:
			# 			self.reward_y = 4 #15

			if self.plume_sampled == True and (self.end_x-4 <= self.rover_x <= self.end_x+4) and (self.end_y-4 <= self.rover_y <= self.end_y+4):
				if self.initial_difference_x >= self.new_difference_x and self.surface_grid[self.rover_x, self.rover_y] != self.TYPE['passed']:
					# Commented lines below are part of an old rewards strategy where the gradient of increasingly positive rewards was larger (up to four grid points from the end point):
					# 
					# if self.rover_x == self.end_x + 4 or self.rover_x == self.end_x - 4:
					# 	self.reward_x = 1 #5
					# if self.rover_x == self.end_x + 3 or self.rover_x == self.end_x - 3:
					# 	self.reward_x = 2 #10
					if self.rover_x == self.end_x + 2 or self.rover_x == self.end_x - 2:
						self.reward_x = 10 # varied over many different values, e.g. 15
					if  self.rover_x == self.end_x + 1 or self.rover_x == self.end_x - 1:
						self.reward_x = 25 # varied over many different values, e.g. 20
				if self.initial_difference_y >= self.new_difference_y and self.surface_grid[self.rover_x, self.rover_y] != self.TYPE['passed']:
					# Commented lines below are part of an old rewards strategy where the gradient of increasingly positive rewards was larger (up to four grid points from the end point):
					#
					# if self.rover_y == self.end_y + 4 or self.rover_y == self.end_y - 4:
					# 	self.reward_y = 1 #5
					# if self.rover_y == self.end_y + 3 or self.rover_y == self.end_y - 3:
					# 	self.reward_y = 2 #10
					if self.rover_y == self.end_y + 2 or self.rover_y == self.end_y - 2:
						self.reward_y = 10 # varied over many different values, e.g. 15
					if self.rover_y == self.end_y + 1 or self.rover_y == self.end_y - 1:
						self.reward_y = 25 # varied over many different values, e.g. 20
			
			# Commented lines below are part of an old rewards strategy where the agent would get a negative reward for passing closely past a ridge or wall:
			#
			# if self.rover_x + 1 == self.TYPE['ridge'] or self.rover_x - 1 == self.TYPE['ridge']:
			# 	self.reward_x = -5
			# if self.rover_y + 1 == self.TYPE['ridge'] or self.rover_y - 1 == self.TYPE['ridge']:
			# 	self.reward_y = -5
			# if self.rover_x == 0 or self.rover_x == self.grid_width - 1:
			# 	self.reward_x = -5
			# if self.rover_y == 0 or self.rover_y == self.grid_height - 1:
			# 	self.reward_y = -5

			self.time_punishment = 0 # originally 1
			
			self.reward = self.reward_x + self.reward_y - self.time_punishment

			if self.surface_grid[self.rover_x, self.rover_y] == self.TYPE['ridge']:
				self.reward = -75 # varied over many different values, e.g. -30, -100 and -200
				done = True

			if self.surface_grid[self.rover_x, self.rover_y] == self.TYPE['plume']:
				self.reward = -50 # varied over many different values, e.g. -30, -50, -100 and -200
				done = True

			if self.surface_grid[self.rover_x, self.rover_y] == self.TYPE['plumesampling'] and self.plume_sampled == False:
				self.reward = 100 # varied over many different values, e.g. 50

			if self.rover_x == self.end_x and self.rover_y == self.end_y and self.plume_sampled == True:
				self.reward = 200 # varied over many different values, e.g. 100
				self.surface_grid[self.end_x, self.end_y] = self.TYPE['rover']
				self.cmap = colors.ListedColormap(['steelblue', 'aquamarine', 'lightgrey', 'black', 'lightskyblue', 'mediumaquamarine'])
				done = True

			if [self.rover_x, self.rover_y] != [self.end_x, self.end_y]:
				self.surface_grid[self.rover_x, self.rover_y] = self.TYPE['rover']

		else:
			self.reward = -100 # varied over many different values, e.g. -150
			done = True

		observations = self.get_observations()

		return observations, self.reward, done, {}, {}
	
	def reset(self, seed=None, options = None):
		if seed:
			np.random.seed(seed)

		self.generate_grid()

		return self.get_observations(), {}
	
if __name__ == "__main__":
	enceladus_environment = EnceladusEnvironment()
	
	# Commented line below was used when first setting up the map.py file to check environment generation and the step function:
	#
	# print(enceladus_environment.step(4)[1])
