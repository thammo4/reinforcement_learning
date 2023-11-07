#
# Implement Value-Iteration solution to determine state-values for Example 6.2: Random Walk
#

import numpy as np;

V = np.array([0, .5, .5, .5, .5, 1.0]);
theta = .0001;

while True:
	V_tmp = V.copy();
	for state in range(1,5):
		#
		# Apply Bellman update equation
		#
		V[state] = .5 * (V[state-1] + V[state+1]);
	if np.max(np.abs(V-V_tmp)) < theta:
		break;

print('Converged state-value function:');
for state, value in zip(['A', 'B', 'C', 'D', 'E'], V[1:]):
    print(f"V({state}) = {value:.4f}")


#
# OUTPUT - Algorithm is not converging to the correct values?
# Should be 1/6, 2/6, 3/6, 4/6, 5/6 for A-E
#

# >>> from value_iteration import *
#
# Converged state-value function:
# V(A) = 0.2000
# V(B) = 0.4000
# V(C) = 0.6000
# V(D) = 0.8000
# V(E) = 1.0000