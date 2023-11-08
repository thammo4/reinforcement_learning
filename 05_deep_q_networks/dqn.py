#
# Implement Deep Q-Network
#

import os, math, pdb, random;
import numpy as np;
import pandas as pd;


import tensorflow as tf;

from keras.models import Sequential;
from keras.layers import Dense, BatchNormalization, LeakyReLU;

from collections import deque;



#
# Define hyperparameters for DRL Agent Problem
#

batch_size = 10;
epochs = 2;
time_steps = 30;


# States
sofa_levels = [0,1,2,3];
num_states = len(sofa_levels);
terminal_state = 3;
state_size = 1; # dimension on state space

# Actions
vaso_dose = [0,1,2,3,4];
num_actions = len(vaso_dose);

print('State Size: ', state_size);
print('Number of Actions: ', num_actions);



class DQN_Agent():
	def __init__ (self, state_levels, state_size, action_size, verbose=False):
		self.state_levels = state_levels;
		self.state_size = state_size;
		self.action_size = action_size;
		self.verbose = verbose;

		self.memory_size = 2000;
		self.gamma = .950;
		self.epsilon = .05;
		self.epsilon_decay = .995;
		self.epsilon_min = .01;
		self.leaky_rate = .01;

		#
		# Define model parameters
		#

		self.learning_rate = .001;
		self.hidden_1_size = 24;
		self.hidden_2_size = 24;

		self.memory = deque(maxlen=self.memory_size);
		self.model = self._build_dqn_model();

	def _build_dqn_model (self):
		model = Sequential();

		#
		# Define Hidden Layer 1
		#

		model.add(Dense(self.hidden_1_size, input_dim=self.state_size));
		model.add(BatchNormalization());
		model.add(LeakyReLU(alpha=self.leaky_rate));

		#
		# Define Hidden Layer 2
		#

		model.add(Dense(self.hidden_2_size));
		model.add(BatchNormalization());
		model.add(LeakyReLU(alpha=self.leaky_rate));

		#
		# Define Output Layer
		#

		model.add(Dense(self.action_size, activation='linear'));
		model.compile(loss='mse', optimizer='adam');

		#
		# Initialize Model by Training on Pairs
		# 	for each state_level: (state_level, random_qvals_per_action)
		#

		for st in self.state_levels:
			model.fit(
				np.array([st]).reshape(1,1),
				np.random.random(self.action_size).reshape(1, self.action_size),
				verbose = 0
			);

		return model;

	def act (self, state):
		'''implement epsilon-greedy action selection'''

	def determine_next_state (self, state, action):
		'''
			Return the next state from the environment.
			Eventually, this should be replaced with simulated/alternative data
		'''

	def compute_reward (self, state):
		'''
			Simple reward function.
			Lower state values are better than larger ones.
		'''

	def momorize (self, state, action, reward, next_state, done):
		'''Cache Transitions'''
		self.memory.append((state, action, reward, next_state, done));

	def replay (self, batch_size):
		'''Fill this in tomorrow!'''