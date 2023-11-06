import numpy as np;

#
# Define relevant constants
#


EPISODE_COUNT = 8000;
DAYS_PER_EPISODE = 20;
ACCT_BAL_0 = 1000;

SHARES_PER_TRADE = 10; # buy/sell fixed number of shares each time for simplicity

ACTIONS = ['hold', 'buy', 'sell'];

#
# Define greek-constants: random-action-rate, learning-rate, discount-rate
#

EPSILON = .05;
ALPHA = .10;
GAMMA = .99;


#
# Initialize state space, action space, and Q-table
#

state_count = 1040;
action_count = 3;

Q = np.zeros((state_count, action_count));



#
# Simulate daily closing price by evaluating f(theta) = 4sin(theta) + 25 for a specified theta (e.g. day)
#

def closing_price (theta):
	return 10*np.sin(theta) + 25;



#
# Define epsilon-greedy policy implementation (e.g. map: states -> actions)
#

def choose_action (state):
	if np.random.rand() < EPSILON:
		return np.random.choice(action_count);
	else:
		return np.argmax(Q[state,:]);


#
# Define function to determine the (next state, reward) when given (current state, action)
#

def execute_action (state, action, shares, bal):
	next_day_price = closing_price(state+1);
	# if action == 'buy':
	if action == 1:
		if bal >= next_day_price * SHARES_PER_TRADE:
			shares += SHARES_PER_TRADE;
			bal -= next_day_price * SHARES_PER_TRADE;
	# elif action == 'sell':
	elif action == 2:
		if shares >= SHARES_PER_TRADE:
			shares -= SHARES_PER_TRADE;
			bal += next_day_price * SHARES_PER_TRADE;

	next_state = (state + 1) % state_count;
	next_acct_value = bal + shares*next_day_price;
	reward = next_acct_value - ACCT_BAL_0;

	return {'next_state':next_state, 'reward':reward, 'shares':shares, 'bal':bal};




#
# Train Agent
#

# theta_t = 0;

for ep in range(EPISODE_COUNT):
	state = ep * DAYS_PER_EPISODE % state_count;
	acct_bal = ACCT_BAL_0;
	shares_held = 0;

	for day in range(DAYS_PER_EPISODE):
		action = choose_action(state);
		action_consequences = execute_action(state, action, shares_held, acct_bal);

		print('state = ', state);
		print('price = ', closing_price(state));

		next_state 	= action_consequences['next_state']; 	print('next state = ', next_state);
		reward 		= action_consequences['reward'];		print('reward = ', reward);
		shares_held = action_consequences['shares']; 		print('shares held = ', shares_held);
		acct_bal 	= action_consequences['bal']; 			print('account balance = ', acct_bal);


		#
		# Define terminal condition (e.g. ran out of money)
		#

		if action_consequences['bal'] <= 0:
			break;


		#
		# Update Q-table per Q-learning update rule
		#

		best_q = np.max(Q[next_state,:]);
		print('update entry for (state, action) = (', state, ', ', action, ') -> ', best_q);
		Q[state, action] += ALPHA * (reward + GAMMA*np.max(Q[next_state,:]) - Q[state,action]);

		state = next_state;
		print('state is now = ', state);
		print('Q\n', Q);

		print('-------------');
		print('\n\n');




#
# Define testing environment for learned agent
#

def test_agent (Q_table, state0, bal0, shares0, state_terminal=(state_count-1)):
	state = state0;
	bal = bal0;
	shares = shares0;
	total_reward = 0;

	while True:
		action = np.argmax(Q_table[state,:]);
		action_consequences = execute_action(state, action, shares, bal);

		next_state = action_consequences['next_state'];
		bal = action_consequences['bal'];
		shares = action_consequences['shares'];
		reward = action_consequences['reward'];

		total_reward += reward;

		state = next_state;

		if bal <= 0 or state == state_terminal:
			break;

	return {'total_reward':total_reward, 'share_held':shares, 'acct_bal':bal};



#
# Test agent
# 	• Iterate over states: state0,...,state_terminal
# 	• At each state, determine action by indexing Q-table row and identifying the column with the largest Q-value (e.g. cumulative discounted total reward)
# 	• Take action -> observe reward and the next state
#



Q_test = test_agent(Q_table=Q, state0=0, bal0=ACCT_BAL_0, shares0=0);
print('------');
print('TESTING\n');
print(Q_test);

