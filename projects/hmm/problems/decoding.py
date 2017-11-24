# Given a HMM Lambda = (A,B), and observation sequence O, determin most probable sequence of hidden states

class Decoder:
	# O is original without adding start-symbol and end-symbol
	@classmethod
	def decode (cls, L, O, endsymbol='</s>', startsymbol='<s>'):
		T = len (O) # Number of observations, including start symbol and end symbol  	
		viterbi = {k:([-1] * (T + 1)) for k in L['A'].keys ()}
		backpointers = {k:([-1] * (T+1)) for k in L['A'].keys ()}

		cls.init_viterbi (L, O, viterbi, backpointers, startsymbol, endsymbol)
		cls.iterate_viterbi (L, O, viterbi, backpointers, startsymbol, endsymbol)
		max_path, max_p = cls.terminate_viterbi (L, O, viterbi, backpointers, startsymbol, endsymbol)
		return max_path, max_p

	@classmethod
	def init_viterbi (cls, L, O, viterbi, backpointers, startsymbol, endsymbol):
		# not initalize start-symbol itself and a transition from start symbol to end symbol.
		t = 1
		for s in L['A'].keys ():
			if s in [startsymbol, endsymbol]: continue
			max_s_prime = cls.estimate_forward_p (viterbi, s, t, L, O, startsymbol, endsymbol)
			cls.set_pointer (s, max_s_prime, t, t-1, backpointers)		

	@classmethod
	def iterate_viterbi (cls, L, O, viterbi, backpointers, startsymbol, endsymbol):
		# select the best path which has most probable sequence, including the end symbol. 
		T = len (O)
		for t in range (2, T + 1):
			for s in L['A'].keys ():
				if s in [startsymbol, endsymbol]: continue
				max_s_prime = cls.estimate_forward_p (viterbi, s, t, L, O, startsymbol, endsymbol)
				cls.set_pointer (s, max_s_prime, t, t-1, backpointers)		
	
	@classmethod
	def terminate_viterbi (cls,L, O, viterbi, backpointers, startsymbol, endsymbol):
		# last_t is the index of end symbol
		last_t = len (O)
		max_s_prime = cls.estimate_forward_p (viterbi, endsymbol, last_t, L, O, startsymbol, endsymbol)
		max_p = viterbi[endsymbol][last_t]
		cls.set_pointer (endsymbol, max_s_prime, last_t, last_t, backpointers)
		max_path = cls.get_max_path (backpointers, viterbi)
		return max_path, max_p

	@staticmethod	
	def set_pointer (s, max_s_prime, t, t_prime, backpointers):
		backpointers[s][t] = (max_s_prime, t_prime)

	@staticmethod	
	def estimate_forward_p (viterbi, s, t, L, O, startsymbol='<s>', endsymbol='</s>'):
		Ot = O[t-1]
		if t - 1 > 0 and s != endsymbol:
			state_list = [i for i in L['A'].keys () if i not in [startsymbol, endsymbol]]
			possible_forward_p = list (map (lambda x: viterbi[x][t-1] * L['A'][s]['cond'][x]['prob'] * L['B'][Ot]['cond'][s]['prob'], state_list))
			max_p = max (possible_forward_p)
			max_trellis_index = possible_forward_p.index (max_p)
			max_s_prime = state_list[max_trellis_index]
			viterbi[s][t] = max_p
		elif t - 1 >= 0 and s == endsymbol:
			max_s_prime = max (viterbi, key=lambda x: viterbi[x][t])
			viterbi[s][t] = viterbi[max_s_prime][t]	
		elif t - 1 == 0:
			max_s_prime = startsymbol
			viterbi[s][t] = L['A'][s]['cond'][max_s_prime]['prob'] * L['B'][Ot]['cond'][s]['prob']
		return max_s_prime

	@staticmethod	
	def get_max_path (pointers, viterbi, endsymbol='</s>', startsymbol='<s>'):
		tnum = len (pointers[endsymbol])
		max_path = []
		cur_s = endsymbol
		cur_t = -1
		for t in range (1, tnum):
			cur_s, cur_t = pointers[cur_s][cur_t]
			max_path.insert (0, cur_s)
		return max_path	




