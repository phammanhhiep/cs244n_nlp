# Given a HMM Lambda = (A,B), and observation sequence O, determin most probable sequence of hidden states

def decode (L, O):
	N = len (L['B'].keys ()) # excluding start and end states
	T = len (L['B'][1])	
	viterbi = [[-1] * (T + 1) for i in range (N + 2)]
	backpointers = [[-1] * (T + 1) for i in range (N + 2)] # the list could store more than one best paths. But in the application, store only one path. Still use 2-dimension list to be consistent with other applications 

	t = 1
	for s in range (1, N + 1):
		max_s_prime = estimate_forward_p (viterbi, s, t, L, O)
		set_pointer (s, max_s_prime, t, backpointers)

	for t in range (2, T + 1):
		for s in range (1, N + 1):
			max_s_prime = estimate_forward_p (viterbi, s, t, L, O)
			set_pointer (s, max_s_prime, t, backpointers)
	
	max_p, max_s_prime = get_max_viterbi (viterbi, t)
	set_pointer (0, max_s_prime, t, backpointers)
	return get_max_path (backpointers, viterbi)

def set_pointer (s, max_s_prime, t, backpointers):
	backpointers[s][t] = (max_s_prime, t)

def estimate_forward_p (viterbi, s, t, L, O):
	max_p, max_s_prime = get_max_viterbi (viterbi, t-1)
	Oi = O[t]
	viterbi[s][t] = max_p * L['A'][max_s_prime][s] * L['B'][s][Oi]
	return max_s_prime 

def get_max_viterbi (viterbi, t):
	max_p, max_s_prime = None, None
	if t != 0:
		max_cell = max (viterbi, key=lambda x: x[t])
		max_s_prime = viterbi.index (max_cell)
		max_p = max_cell[t]
	else:
		max_p = 1
		max_s_prime = 0
	return max_p, max_s_prime		

def get_max_path (pointers, viterbi):
	tnum = len (pointers[0])
	last_s, last_t = pointers[0][-1]
	max_path = []
	p = 1
	cur_s = last_s
	cur_t = last_t

	for t in range (1, tnum-1):
		p *= viterbi[cur_s][cur_t]
		max_path.insert (0, cur_s)
		cur_s, cur_t = pointers[cur_s][cur_t]
	max_path.insert (0, 0) # the start state
	return max_path, p




