import random;
import numpy as np;

# Define States
price_buckets = [0, 50, 100, 150];
num_states = len(price_buckets); terminal_state = 150;

# Define Actions {0:Hold, 1:Buy, 2:Sell}
actions = [0, 1, 2];
num_actions = len(actions);


# Initialize Q-function with zeros
Q = np.zeros(shape=(num_states, num_actions));

# Epsilon-greedy exploration
def act(epsilon, action_values):
	action_size = len(action_values);
	if np.random.rand() <= epsilon:
		return random.randrange(action_size);
	return np.argmax(action_values);


def calc_reward (state, action, next_state):
	if action == 1:
		return next_state - state;
	elif action == 2:
		return state - next_state;
	else:
		return 0;

def determine_next_state (state, action):
	if action == 1:
		next_state = min(terminal_state, state+50);
	elif action == 2:
		next_state = max(0, state-50);
	else:
		print('random state!!');
		next_state = random.choice(price_buckets);
	return next_state;





num_episodes = 5000;
max_timesteps = 100;

epsilon=.10; alpha=.10; gamma=.99

# for ep in range(num_episodes):
for ep in range(1):
	if ep % 10 == 0:
		print('Episode = ', ep);
	print('(State, Action, Reward, Next State) Transition Probabilities');

	current_price = 0;
	# done = False;

	for tm in range(max_timesteps):
		action = act(epsilon, Q[current_price, :]);
		next_price = determine_next_state(current_price, action);
		reward = calc_reward(current_price, action, next_price);

		print('current price = ', current_price);
		print('action =', action);
		print('reward = ', reward);
		print('next price = ', next_price);
		

		transition = (current_price, action, reward, next_price);
		print('(S_t, A_t, R_t+1, S_t+1) = ', transition);

		# Update Q(s,a) per TD(0)
		Q[current_price, action] += alpha * (reward + gamma*np.amax(Q[next_price,:]) - Q[current_price,action]);

		current_price = next_price;

		if next_price == terminal_state:
			break; # done = True;

		if ep & 10 == 0:
			print('Q\n', Q);

		

		print('\n');



# for ep in range(num_episodes):
# 	if ep % 5 == 0:
# 		print('Episode =', ep+1)
# 	print('(State, Action, Reward, Next State) Transitions');

# 	current_price=0; done=False;

# 	for tm in range(max_timesteps):
# 		action = act(epsilon, Q[current_price, :]);
# 		next_price = determine_next_state(current_price, action);
# 		reward = calc_reward(current_price, action, next_price);

# 		transition = (current_price, action, reward, next_price); print(transition);
# 		# Q[current_price, action] += alpha*(reward + gamma*np.amax(Q[next_price,:])-Q[current_price,action]);

# 		# Q[current_price, action] += alpha*(reward + gamma*np.amax(Q[next_price,:])-Q([current_price,:]));
# 		# Q[current_price, action] += alpha*(reward + gamma*np.amax(Q[next_price,:])-Q([current_price,:]) f)


# 		current_price = next_price;
# 		if next_price == terminal_state:
# 			break;
# 		if ep % 10 == 0:
# 			print('Q\n', Q);


