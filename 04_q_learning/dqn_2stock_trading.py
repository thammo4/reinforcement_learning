import time; start_time = time.perf_counter();

import os, dotenv;
import numpy as np;

from tensorflow.keras.models import Sequential;
from tensorflow.keras.layers import Dense;
from tensorflow.keras.optimizers import Adam;



#
# Define relevant constants
#

EPISODE_COUNT = 3750; # 7500;
DAYS_PER_EPISODE = 75;
ACCT_BAL_0 = 1000;

SHARES_PER_TRADE = 10;

actions = ('hold', 'buy', 'sell');
ACTIONS = [(a,b) for a in actions for b in actions];


#
# Define greek hyperparameters: random action rate, learn rate, discount rate
#

EPSILON = .125;
ALPHA = .20;
GAMMA = .950;


#
# Define state-space/action-space size
#

# state_count_A, state_count_B = 750, 750;
state_count_A, state_count_B = 375, 375;
action_countA, action_count_B = 3, 3;


#
# Simulate daily closing price by evaluating f(theta) = 4sin(theta) + 25 for a specified theta (e.g. day)
# Stock XYZ
#

def closing_price_A (theta):
	return 12*np.sin(.3*theta) + 25;


#
# Simulate daily closing price by evaluating f(theta) = 25cos(.5*theta) + 35 for specified theta (e.g day)
#

def closing_price_B (theta):
	return 25*np.cos(.5*theta) + 35;


#
# Define Feed-Forward ANN containing a single hidden layer with nonlinear ReLU activation functions
#

def q_ann (input_size, output_size, hidden_layer_size=64):
	model = Sequential();
	model.add(Dense(hidden_layer_size, input_dim=input_size, activation='relu'));
	model.add(Dense(output_size, activation='linear'));
	model.compile(loss='mse', optimizer=Adam(lr=ALPHA));

	return model;



#
# One-hot encode states
#

def one_hot_states (state_A, state_B):
	state = np.zeros(state_count_A+state_count_B);
	state[state_A] = 1;
	state[state_count_A + state_B] = 1;
	return state;


#
# Define function to determine the (next state, reward) when given (current state, action)
#

def execute_action (state_A, state_B, action_idx, shares_A, shares_B, bal):
	action_A, action_B = ACTIONS[action_idx];

	#
	# Define action for Stock A
	#

	next_day_price_A = closing_price_A(state_A + 1);
	next_day_price_B = closing_price_B(state_B + 1);

	if action_A == 'buy':
		if bal >= next_day_price_A * SHARES_PER_TRADE:
			shares_A += SHARES_PER_TRADE;
			bal -= next_day_price_A * SHARES_PER_TRADE;
	elif action_A == 'sell':
		if shares_A >= SHARES_PER_TRADE:
			shares_A -= SHARES_PER_TRADE;
			bal += next_day_price_A * SHARES_PER_TRADE;

	#
	# Define action for Stock B
	#

	if action_B == 'buy':
		if bal >= next_day_price_B * SHARES_PER_TRADE:
			shares_B += SHARES_PER_TRADE;
			bal -= next_day_price_B  * SHARES_PER_TRADE;
	elif action_B == 'sell':
		if shares_B >= SHARES_PER_TRADE:
			shares_B -= SHARES_PER_TRADE;
			bal += next_day_price_B * SHARES_PER_TRADE;

	#
	# Define next state for Stocks A, B
	#

	next_state_A = (state_A +1) % state_count_A;
	next_state_B = (state_B +1) % state_count_B;

	#
	# Account value is weighted combination of shares and price
	#

	next_acct_value = bal + shares_A*next_day_price_A + shares_B*next_day_price_B;

	#
	# Compute reward relative to starting balance
	#

	reward = next_acct_value - ACCT_BAL_0;

	return {
		'next_state_A' 	: next_state_A,
		'next_state_B' 	: next_state_B,
		'shares_A' 		: shares_A,
		'shares_B' 		: shares_B,
		'bal' 			: bal,
		'reward' 		: reward
	};

#
# Define neural network I/O dimensions
#

ann_input_size = state_count_A + state_count_B;
ann_output_size = len(ACTIONS);

q_network = q_ann(ann_input_size, ann_output_size);



#
# Define epsilon-greedy policy implementation (e.g. map: states -> actions)
#

def choose_action (state_A, state_B, q_net):
	if np.random.rand() < EPSILON:
		return np.random.choice(ann_output_size);
	else:
		state = one_hot_states(state_A, state_B);
		q_vals = q_net.predict(state.reshape(1, -1));

	return np.argmax(q_vals[0]);


for ep in range(EPISODE_COUNT):
	state_A = ep * DAYS_PER_EPISODE % state_count_A;
	state_B = ep * DAYS_PER_EPISODE % state_count_B;

	acct_bal = ACCT_BAL_0;

	shares_held_A = 0;
	shares_held_B = 0;

	for day in range(DAYS_PER_EPISODE):

		#
		# Represent state with one-hot encoded value
		#

		state = one_hot_states(state_A, state_B);

		#
		# Choose action per epsilon-greedy policy
		#

		action_index = choose_action(state_A, state_B, q_network);

		action_outcome = execute_action(state_A, state_B, action_index, shares_held_A, shares_held_B, acct_bal);

		print('priceA = ', closing_price_A(state_A));
		print('priceB = ', closing_price_B(state_B));

		next_state_A = action_outcome['next_state_A']; 	print('nextA = ', next_state_A);
		next_state_B = action_outcome['next_state_B']; 	print('nextB = ', next_state_B);
		shares_held_A = action_outcome['shares_A']; 	print('sharesA = ', shares_held_A);
		shares_held_B = action_outcome['shares_B']; 	print('sharesB = ', shares_held_B);
		reward = action_outcome['reward']; 				print('reward = ', reward);
		acct_bal = action_outcome['bal']; 				print('account balance = ', acct_bal);


		#
		# Compute target weights
		#

		next_state 	= one_hot_states(next_state_A, next_state_B);
		q_vals_next = q_network.predict(next_state.reshape(1,-1));
		target 		= reward + GAMMA * np.amax(q_vals_next[0]);


		#
		# Update target weights
		#

		q_vals = q_network.predict(state.reshape(1, -1));
		q_vals[0][action_index] = target;
		q_network.fit(state.reshape(1,-1), q_vals, epochs=1, verbose=0);


		#
		# Define terminal condition (e.g. ran out of money)
		#

		if acct_bal <= 0:
			print('!!!!!! BANKRUPT !!!!!!!!');
			break;




		#
		# Update Q-table per Q-learning update rule
		#

		# print('update entry for (state_A, state_B, action) = (', state_A, ', ', state_B, ', ', action_index, ') -> ', np.max(Q[next_state_A, next_state_B, :]));

		# Q[state_A, state_B, action_index] += ALPHA * (reward + GAMMA * np.max(Q[next_state_A, next_state_B, :]) - Q[state_A, state_B, action_index]);

		state_A, state_B = next_state_A, next_state_B;
		# print('Q\n', Q);
		print('Q VALUES\n', q_vals);

		print('-------------');
		print('\n');




#
# Define testing environment for learned agent
#

def test_agent (q_net, state0_A, state0_B, shares0_A, shares0_B, bal0, max_iterations=1e4):
	state_A, state_B 	= state0_A, state0_B;
	shares_A, shares_B 	= shares0_A, shares0_B;

	bal = bal0;
	total_reward = 0;

	iter = 0;
	while True:

		state = one_hot_states(state_A, state_B);

		q_vals = q_net.predict(state.reshape(1,-1));

		action = np.argmax(q_vals[0]);
		action_outcome 	= execute_action(state_A, state_B, action, shares_A, shares_B, bal);

		next_state_A 	= action_outcome['next_state_A'];
		shares_A 		= action_outcome['shares_A'];

		next_state_B 	= action_outcome['next_state_B'];
		shares_B 		= action_outcome['shares_B'];

		bal 	= action_outcome['bal'];
		reward 	= action_outcome['reward'];

		total_reward += reward;

		state_A, state_B = next_state_A, next_state_B;

		#
		# Terminal condition for intra-episode iterations
		#

		if bal <= 0 or iter == int(max_iterations):
			break;

		iter += 1;

	return {'total_reward':total_reward, 'acct_bal':bal, 'shares':(shares_A, shares_B)};




#
# Test agent
# 	• Iterate over states: state0,...,state_terminal
# 	• At each state, determine action by evaluating the ANN-approximated q-function
# 	• Take action -> observe reward and the next state
#

Q_test = test_agent(q_net=q_network, state0_A=0, state0_B=0, shares0_A=0, shares0_B=0, bal0=ACCT_BAL_0);
print('----------');
print('TESTING\n'); print(Q_test, '\n');


end_time = time.perf_counter();
elapsed_time = end_time - start_time;
print(f"Program took {elapsed_time} seconds.");














































































































































