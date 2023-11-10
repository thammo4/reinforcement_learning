import time; start_time = time.perf_counter();

import os, dotenv;
import numpy as np;

import yfinance as yf;

from uvatradier import Tradier, Quotes;

dotenv.load_dotenv();

#
# Authenticate Tradier API
#

tradier_acct 	= os.getenv('tradier_acct');
tradier_token 	= os.getenv('tradier_token');


#
# Instantiate Quotes object and fetch 500 bars of SPY data
#

# >>> SPY
#
#            date    open     high       low   close     volume
# 4    2021-11-05  469.28  470.650  466.9200  468.53   66390563
# 5    2021-11-08  469.70  470.230  468.2031  468.93   50405194
# 6    2021-11-09  469.32  469.570  465.8800  467.38   51149147
# 7    2021-11-10  465.58  467.380  462.0400  463.62   69429653
# 8    2021-11-11  465.21  465.290  463.7500  463.77   34848495
# ..          ...     ...      ...       ...     ...        ...
# 499  2023-10-26  416.45  417.325  411.6000  412.55  115156761
# 500  2023-10-27  414.19  414.600  409.2100  410.68  107367671
# 501  2023-10-30  413.56  416.680  412.2200  415.59   86562675
# 502  2023-10-31  416.18  418.530  414.2100  418.20   79665150
# 503  2023-11-01  419.20  423.500  418.6499  422.66   98068115
#
# [500 rows x 6 columns]

quotes = Quotes(tradier_acct, tradier_token);
SPY = quotes.get_historical_quotes(symbol='SPY', start_date='2021-11-04', end_date='2023-10-31')[4:]; # [4:] because SPY returns 504 rows, but we only need 500


#
# Fetch 500 bars of daily ^VIX data
# Note - yahoo finance does not include the `end` date in the returned dataframe
#

# >>> VIX
#                                 Open       High        Low      Close  Volume  Dividends  Stock Splits
# Date
# 2021-11-04 00:00:00-05:00  15.060000  16.139999  14.730000  15.440000       0        0.0           0.0
# 2021-11-05 00:00:00-05:00  15.590000  17.020000  14.950000  16.480000       0        0.0           0.0
# 2021-11-08 00:00:00-06:00  17.230000  17.690001  16.440001  17.219999       0        0.0           0.0
# 2021-11-09 00:00:00-06:00  17.430000  18.570000  17.209999  17.780001       0        0.0           0.0
# 2021-11-10 00:00:00-06:00  17.740000  19.900000  17.219999  18.730000       0        0.0           0.0
# ...                              ...        ...        ...        ...     ...        ...           ...
# 2023-10-25 00:00:00-05:00  19.389999  21.240000  18.860001  20.190001       0        0.0           0.0
# 2023-10-26 00:00:00-05:00  21.780001  21.959999  20.219999  20.680000       0        0.0           0.0
# 2023-10-27 00:00:00-05:00  20.389999  22.070000  19.719999  21.270000       0        0.0           0.0
# 2023-10-30 00:00:00-05:00  21.129999  21.160000  19.549999  19.750000       0        0.0           0.0
# 2023-10-31 00:00:00-05:00  19.860001  19.860001  17.969999  18.139999       0        0.0           0.0
#
# [500 rows x 7 columns]

VIX = yf.Ticker('^VIX').history(start='2021-11-04', end='2023-11-01', interval='1d');



#
# Define relevant constants
#


EPISODE_COUNT = 3750;
DAYS_PER_EPISODE = 50;
ACCT_BAL_0 = 1000;

SHARES_PER_TRADE = 10; # buy/sell fixed number of shares each time for simplicity

ACTIONS = ['hold', 'buy', 'sell'];

#
# Define greek-constants: random-action-rate, learning-rate, discount-rate
#

EPSILON = .125;
ALPHA = .175;
GAMMA = .90;


#
# Initialize state space, action space, and Q-table
#

# state_count = 5000;
state_count = 750;
action_count = 3;

Q = np.zeros((state_count, action_count));



#
# Simulate daily closing price by evaluating f(theta) = 4sin(theta) + 25 for a specified theta (e.g. day)
#

def closing_price (theta):
	return 12*np.sin(.3*theta) + 25;


#
# Define epsilon-greedy policy implementation (e.g. map: states -> actions)
#

def choose_action (state):
	if np.random.rand() < EPSILON:
		print('!!!!!!!!!!!!!!!!!!!!!!!! RANDOM ACTION !!!!!!!!!!!!!!!!!!!!!!!!!!');
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
print(Q_test, '\n');

end_time = time.perf_counter();
elapsed_time = end_time - start_time;
print(f"Program executed in {elapsed_time} seconds");