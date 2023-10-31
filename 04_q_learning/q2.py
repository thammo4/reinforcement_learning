import numpy as np;
import pandas as pd;
import random;
import os;
import dotenv;

from uvatradier import Tradier, Quotes;

dotenv.load_dotenv();

tradier_acct 	= os.getenv('tradier_acct');
tradier_token 	= os.getenv('tradier_token');


quotes = Quotes(tradier_acct, tradier_token);

csx = quotes.get_historical_quotes(symbol='CSX', interval='daily', start_date='2022-10-01', end_date='2023-10-01');

action_space = ['buy', 'sell', 'hold'];
state_space = list(csx['close']);

num_states = len(state_space);



# csx_close = list(csx['close']); 

def action (day_index, action_space, epsilon):
	if np.random.rand() <= epsilon:
		return random.choice(action_space);
	if day_index % 5 == 0:
		return action_space[1]; # sell every 5th day
	return action_space[0];



def calc_reward (state):
	print('state: ', state);

# def determine_next_state (current_state, action_taken):
# 	print('current state: ', current_state);
# 	print('action taken: ', action_taken);

# def determine_next_state (day_index):
# 	return state_space[day_index];

# def determine_next_state (current_price, acct_bal, portfolio_val, num_holdings, action):
# 	if action == 'buy':
# 		acct_bal -= current_price;
# 		num_holdings += 1;
# 	if action == 'sell':
# 		acct_bal += current_price*num_holdings;
# 		num_holdings = 0;

# 	portfolio_val = acct_bal + current_price * num_holdings;
# 	return (acct_bal, portfolio_val, num_holdings);


def determine_next_state (current_price, acct_bal, portfolio_val, num_holdings, action):
	if action == 'buy':
		acct_bal -= current_price;
		num_holdings += 1;
	elif action == 'sell':
		acct_bal += current_price * num_holdings;
		num_holdings = 0;

	portfolio_val = acct_bal + num_holdings * current_price;

	return {'account_balance':acct_bal, 'portfolio_value':portfolio_val, 'num_holdings':num_holdings};









current_balance = 100;







