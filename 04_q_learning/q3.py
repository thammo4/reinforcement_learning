import pandas as pd
import numpy as np

import os, dotenv;
from uvatradier import Tradier, Quotes;

dotenv.load_dotenv();

tradier_acct = os.getenv("tradier_acct"); tradier_token = os.getenv("tradier_token");

quotes = Quotes(tradier_acct, tradier_token);

csx = quotes.get_historical_quotes(symbol='CSX', interval='daily', start_date='2022-10-02', end_date='2023-09-29');

# Define parameters
alpha = 0.1  # learning rate
gamma = 0.95  # discount factor
epsilon = 0.1  # for epsilon-greedy exploration

# Initialize the Q-table. 
# Rows are states (close prices), columns are actions (buy, sell, hold)
# Here we'll initialize it to have one row for every unique close price and 3 columns for the actions.
unique_prices = csx['close'].unique()
Q = pd.DataFrame(0, index=unique_prices, columns=['buy', 'sell', 'hold'])

# Define the reward function
def get_reward(current_price, next_price, action):
    if action == 'buy':
        return next_price - current_price
    elif action == 'sell':
        return current_price
    else:
        return 0

# The policy for action selection
def epsilon_greedy_policy(state):
    if np.random.rand() < epsilon:
        return np.random.choice(['buy', 'sell', 'hold'])
    else:
        return Q.loc[state].idxmax()

# Train using TD Learning
for week_start in range(0, len(csx)-5, 5):  # assuming 5 trading days in a week
    print('WEEK: ', week_start);
    owned_shares = 0
    for day in range(week_start, week_start + 4):  # iterate over the 5 days in the week
        current_price = csx['close'].iloc[day]
        next_price = csx['close'].iloc[day+1]

        action = epsilon_greedy_policy(current_price)
        
        if action == 'sell':
            owned_shares = 0
        elif action == 'buy':
            owned_shares += 1

        reward = get_reward(current_price, next_price, action)
        
        # TD update
        best_next_action    = Q.loc[next_price].idxmax()
        td_target           = reward + gamma * Q.loc[next_price, best_next_action]
        td_error            = td_target - Q.loc[current_price, action]
        Q.loc[current_price, action] += alpha * td_error

        print('Q-day\n', Q.T); print('\n');

print(Q)
