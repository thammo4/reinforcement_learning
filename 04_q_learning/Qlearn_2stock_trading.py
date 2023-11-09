import time; start_time = time.perf_counter();

import os, dotenv;
import numpy as np;


# import yfinance as yf;

# from uvatradier import Tradier, Quotes;

# dotenv.load_dotenv();

# #
# # Authenticate Tradier API
# #

# tradier_acct 	= os.getenv('tradier_acct');
# tradier_token 	= os.getenv('tradier_token');

#
# Define relevant constants
#


# EPISODE_COUNT 		= 3750;
# DAYS_PER_EPISODE 	= 50;
# EPISODE_COUNT 		= 2500;
# DAYS_PER_EPISODE 	= 25;
# EPISODE_COUNT 		= 2750;
# DAYS_PER_EPISODE 	= 50;

# EPISODE_COUNT = 3750;
# DAYS_PER_EPISODE = 25;

EPISODE_COUNT = 5000;
DAYS_PER_EPISODE = 50;
ACCT_BAL_0 			= 1000;

SHARES_PER_TRADE = 10; # buy/sell fixed number of shares each time for simplicity

# ACTIONS = ['hold', 'buy', 'sell'];
actions = ('hold', 'buy', 'sell');
ACTIONS = [(a,b) for a in actions for b in actions];

#
# Define greek-constants: random-action-rate, learning-rate, discount-rate
#

EPSILON = .125;
ALPHA = .15;
GAMMA = .875;


#
# Initialize state space, action space, and Q-table
#

# state_count_A = 5000;
# state_count_A = 750;
# state_count_B = 750;

# state_count_A = 250;
# state_count_B = 250;
state_count_A = 350;
state_count_B = 350;

action_count_A = 3;
action_count_B = 3;

action_count = action_count_A * action_count_B;

Q = np.zeros((state_count_A, state_count_B, action_count));



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
# Define epsilon-greedy policy implementation (e.g. map: states -> actions)
#

# def choose_action (state):
# 	if np.random.rand() < EPSILON:
# 		print('!!!!!!!!!!!!!!!!!!!!!!!! RANDOM ACTION !!!!!!!!!!!!!!!!!!!!!!!!!!');
# 		return np.random.choice(action_count_A);
# 	else:
# 		return np.argmax(Q[state,:]);

def choose_action (state_A, state_B):
	if np.random.rand() < EPSILON:
		print('RANDOM');
		return np.random.choice(len(ACTIONS));
	else:
		return np.argmax(Q[state_A, state_B, :]);



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

	next_state_A = (state_A +1) % state_count_A;
	next_state_B = (state_B +1) % state_count_B;


	#
	# Account value is weighted combination of shares and price
	#

	next_acct_value = bal + shares_A*next_day_price_A + shares_B*next_day_price_B;

	#
	# Reward computed relative to starting balance
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
# Train Agent
#

for ep in range(EPISODE_COUNT):
	# state = ep * DAYS_PER_EPISODE % state_count_A*state_count_B;
	state_A = ep * DAYS_PER_EPISODE % state_count_A;
	state_B = ep * DAYS_PER_EPISODE % state_count_B;

	acct_bal = ACCT_BAL_0;

	# shares_held = 0;
	shares_held_A = 0;
	shares_held_B = 0;

	for day in range(DAYS_PER_EPISODE):
		# action = choose_action(state);
		# action_consequences = execute_action(state, action, shares_held, acct_bal);
		action_index = choose_action(state_A, state_B);
		action_outcome = execute_action(state_A, state_B, action_index, shares_held_A, shares_held_B, acct_bal);

		# print('state = ', state);
		print('priceA = ', closing_price_A(state_A));
		print('priceB = ', closing_price_B(state_B));

		next_state_A = action_outcome['next_state_A']; 	print('nextA = ', next_state_A);
		next_state_B = action_outcome['next_state_B']; 	print('nextB = ', next_state_B);
		shares_held_A = action_outcome['shares_A']; 	print('sharesA = ', shares_held_A);
		shares_held_B = action_outcome['shares_B']; 	print('sharesB = ', shares_held_B);
		reward = action_outcome['reward']; 				print('reward = ', reward);
		acct_bal = action_outcome['bal']; 				print('account balance = ', acct_bal);

		# next_state 	= action_consequences['next_state']; 	print('next state = ', next_state);
		# reward 		= action_consequences['reward'];		print('reward = ', reward);
		# shares_held = action_consequences['shares']; 		print('shares held = ', shares_held);
		# acct_bal 	= action_consequences['bal']; 			print('account balance = ', acct_bal);


		#
		# Define terminal condition (e.g. ran out of money)
		#

		# if action_consequences['bal'] <= 0:
		if acct_bal <= 0:
			print('!!!!!! BANKRUPT WTF ??????????');
			break;


		#
		# Update Q-table per Q-learning update rule
		#

		print('update entry for (state_A, state_B, action) = (', state_A, ', ', state_B, ', ', action_index, ') -> ', np.max(Q[next_state_A, next_state_B, :]));

		Q[state_A, state_B, action_index] += ALPHA * (reward + GAMMA * np.max(Q[next_state_A, next_state_B, :]) - Q[state_A, state_B, action_index]);

		# state = next_state;
		state_A, state_B = next_state_A, next_state_B;
		# print('state is now = ', state);
		print('Q\n', Q);

		print('-------------');
		print('\n');




#
# Define testing environment for learned agent
#

# def test_agent (Q_table, state0_A, state0_B, shares0_A, shares0_B, bal0, state_terminal=(-1+state_count_A*state_count_B)):
def test_agent (Q_table, state0_A, state0_B, shares0_A, shares0_B, bal0, max_iterations=1e4):
	state_A, state_B = state0_A, state0_B;
	shares_A, shares_B = shares0_A, shares0_B;

	bal = bal0;
	total_reward = 0;

	iter = 0;
	while True:
		action = np.argmax(Q_table[state_A, state_B, :]);
		action_outcome = execute_action(state_A, state_B, action, shares_A, shares_B, bal);

		next_state_A = action_outcome['next_state_A'];
		shares_A = action_outcome['shares_A'];

		next_state_B = action_outcome['next_state_B'];
		shares_B = action_outcome['shares_B'];

		bal = action_outcome['bal'];
		reward = action_outcome['reward'];

		total_reward += reward;

		state_A, state_B = next_state_A, next_state_B;

		# if bal <= 0 or state_A*state_B == state_terminal:
		if bal <= 0 or iter == int(max_iterations):
			break;

		iter += 1;

	return {'total_reward':total_reward, 'acct_bal':bal, 'shares':(shares_A, shares_B)};

#
# Test agent
# 	• Iterate over states: state0,...,state_terminal
# 	• At each state, determine action by indexing Q-table row and identifying the column with the largest Q-value (e.g. cumulative discounted total reward)
# 	• Take action -> observe reward and the next state
#

# def test_agent (Q_table, state0_A, state0_B, shares0_A, shares0_B, bal0, max_iterations=1e4):

Q_test = test_agent(Q_table=Q, state0_A=0, state0_B=0, shares0_A=0, shares0_B=0, bal0=ACCT_BAL_0);
print('----------');
print('TESTING\n');
end_time = time.perf_counter();
elapsed_time = end_time - start_time;
print(f"Program executed in {elapsed_time} seconds");

print(Q_test);



# Q_test = test_agent(Q_table=Q, state0=0, bal0=ACCT_BAL_0, shares0=0);
# print('------');
# print('TESTING\n');
# print(Q_test);