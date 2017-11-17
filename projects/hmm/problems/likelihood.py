# Given a HMM Lambda = (A,B), and observation sequence O, determine likelihood P (O|Lambda)
# A is a set of transition probabilities; B is a set of emmission probabilities

def estimate_likelihood (L, O):
	N = len (L['A'].keys ()) - 2 # number of states excluding start and end
	T = len (L['B'][1]) # number of observations
	forward = [[None] * (T + 1) for i in range (N + 2)]
	Oi = O[0]
	for s in range (1, N + 1):
		forward[s][1] = L['A'][0][s] * L['B'][s][Oi]
	for t in range (2, T + 1):
		for s in range (1, N + 1):
			forward[s][t] = sum_forward (forward, L, O, t, s)
	return total_p (forward, t), forward		

def total_p (forward, t):
	total = [forward[i][t] for i in range (len (forward) - 2) if forward[i][t] is not None]
	return sum (total)

def sum_forward (forward, L, O, t, s):
	N = len (L['A'].keys ()) - 2
	total = 0
	Oi = O[t]
	for s_prime in range (1, N + 1):
		total += forward[s_prime][t-1] * L['A'][s_prime][s] * L['B'][s][Oi]
	return total	

