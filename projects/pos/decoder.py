from itertools import product
import os, sys
sys.path.insert (0, '\\'.join(os.getcwd ().split ('\\')[:-1]))
import hmm.problems.decoding as dcd
HMM_Decoder = dcd.Decoder

class Decoder (HMM_Decoder):
	# assume observation include start symbols and end symbol. The number of such symbols match the n-gram. Like with 2-gram, there is 1 end and 1 start. With 3-gram, there are 2 starts and 1 end. 
	def __init__ (self): pass

	@classmethod
	def decode (cls, L, O, ngram=2, endsymbol='</s>', startsymbol='<s>'):
		T = len (O) # Number of observations, including start symbol and end symbol  	
		viterbi = {k:([-1] * T) for k in L['A'].keys () if len (k.split (' ')) == 1}
		backpointers = {k:([-1] * T) for k in L['A'].keys () if len (k.split (' ')) == 1}
		cls.init_viterbi (L, O, viterbi, backpointers, startsymbol, endsymbol, ngram=ngram)
		cls.iterate_viterbi (L, O, viterbi, backpointers, startsymbol, endsymbol, ngram=ngram)
		max_p = cls.terminate_viterbi (L, O, viterbi, backpointers, startsymbol, endsymbol, ngram=ngram)
		max_path = cls.get_max_path (backpointers, ngram=ngram)
		return max_path, max_p

	@classmethod
	def init_viterbi (cls, L, O, viterbi, backpointers, startsymbol, endsymbol, ngram=2):
		# not initalize start-symbol itself and a transition from start symbol to end symbol.
		t = ngram - 1
		states = [k for k in L['A'].keys () if len (k.split (' ')) == 1]
		for s in states:
			if s in [startsymbol, endsymbol]: continue
			max_s_prime = cls.estimate_forward_p (viterbi, backpointers, s, t, L, O, ngram, startsymbol, endsymbol)
			cls.set_pointer (s, max_s_prime, t, backpointers)		

	@classmethod
	def iterate_viterbi (cls, L, O, viterbi, backpointers, startsymbol, endsymbol, ngram=2):
		# select the best path which has most probable sequence, including the end symbol. 
		T = len (O)
		for t in range (ngram, T-1): # not consider the last word (end symbol)
			states = [k for k in L['A'].keys () if len (k.split (' ')) == 1]
			for s in states:
				if s in [startsymbol, endsymbol]: continue			
				max_s_prime = cls.estimate_forward_p (viterbi, backpointers, s, t, L, O, ngram, startsymbol, endsymbol)
				cls.set_pointer (s, max_s_prime, t, backpointers)

	@classmethod
	def terminate_viterbi (cls,L, O, viterbi, backpointers, startsymbol, endsymbol, ngram=2):
		# last_t is the index of end symbol
		last_t = len (O) - 1
		max_s_prime = cls.estimate_forward_p (viterbi, backpointers, endsymbol, last_t, L, O, ngram, startsymbol, endsymbol)
		max_p = viterbi[endsymbol][last_t]
		cls.set_pointer (endsymbol, max_s_prime, last_t, backpointers)
		return max_p

	@staticmethod	
	def set_pointer (s, max_s_prime, t, backpointers):
		# relate to the approach to find the best path. If change estimate_forward_p, need to reconsider the implementation 
		backpointers[s][t] = max_s_prime

	@staticmethod
	def get_prev_states (backpointers, t, ngram=2, startsymbol='<s>', endsymbol='</s>'):
		temp_states = []
		temp = []
		states = []
		for i in range (ngram-1):
			if i == 0:
				temp = [si for si,v in backpointers.items () if si not in [startsymbol, endsymbol]]
			else:
				temp = [backpointers[si][t-i] for si in temp]	
			temp_states.insert (0, temp)	
		for i in zip (*temp_states):
			states.append (' '.join (i))
		return states

	@classmethod
	def estimate_forward_p (cls, viterbi, backpointers, s, t, L, O, ngram=2, startsymbol='<s>', endsymbol='</s>'):
		# Only consider tags in the partial sequences
		# Easy solution for tracing back
		Ot = O[t]
		if t-ngram+1 > 0:
			prev_states = cls.get_prev_states (backpointers, t, ngram)
			possible_forward_p = list (map (lambda x: viterbi[x.split (' ')[-1]][t-1] * L['A'][s]['cond'][x]['prob'] * L['B'][Ot]['cond'][s]['prob'], prev_states))
			max_p = max (possible_forward_p)
			max_trellis_index = possible_forward_p.index (max_p)
			max_s_prime = prev_states[max_trellis_index]
			viterbi[s][t] = max_p			
		elif t-ngram+1 == 0:
			max_s_prime = ' '.join([startsymbol] * (ngram - 1))
			viterbi[s][t] = L['A'][s]['cond'][max_s_prime]['prob'] * L['B'][Ot]['cond'][s]['prob']
		max_s_prime = max_s_prime.split (' ')[-1]	
		return max_s_prime	

	@staticmethod	
	def get_max_path (pointers, ngram=2, endsymbol='</s>', startsymbol='<s>'):
		tnum = len (pointers[endsymbol])
		max_path = []
		cur_s = endsymbol
		for t in range (1, tnum-ngram+1):
			cur_t = tnum - t
			cur_s = pointers[cur_s][cur_t]
			max_path.insert (0, cur_s)
		return max_path

	# @staticmethod
	# def gen_combined_t_primes (L, O, t, ngram=2):
	# 	# t: index of the current word
	#	# no longer use. change approach to implement estimate_forward_p, and so this implementation has no use.
	# 	t_primes = []
	# 	nw = O[t-ngram+1:t] # a list of (ngram - 1) previous words of the word in question
	# 	nw_num = len (nw)
	# 	for j in range (nw_num):
	# 		temp_t = list (L['B'][nw[j]]['cond'].keys ())
	# 		t_primes.append (temp_t)
	# 	combined_t = [i for i in product (*t_primes)]
	# 	return combined_t	

	# @classmethod
	# def estimate_forward_p_2 (cls, viterbi, backpointers, s, t, L, O, ngram=2, startsymbol='<s>', endsymbol='</s>'):
	# 	# the implementation has no easy traceback solution
	# 	def _multiple_prev_viterbi (viterbi, x, t):
	# 		p = 1
	# 		x = x.split (' ')
	# 		num = len (x)
	# 		for i in range (num):
	# 			state = x[i]
	# 			p *= viterbi[state][i +t-num]
	# 		return p

	# 	Ot = O[t]
	# 	if t-ngram+1 > 0:
	# 		prev_state_list = cls.gen_combined_t_primes (L, O, t, ngram)
	# 		possible_forward_p = list (map (lambda x: _multiple_prev_viterbi (viterbi, x, t) * L['A'][s]['cond'][x]['prob'] * L['B'][Ot]['cond'][s]['prob'], prev_state_list))
	# 		max_p = max (possible_forward_p)
	# 		max_trellis_index = possible_forward_p.index (max_p)
	# 		max_s_prime = ' '.join (prev_state_list[max_trellis_index])
	# 		viterbi[s][t] = max_p			
	# 	elif t-ngram+1 == 0:
	# 		max_s_prime = ' '.join([startsymbol] * (ngram - 1))
	# 		viterbi[s][t] = L['A'][s]['cond'][max_s_prime]['prob'] * L['B'][Ot]['cond'][s]['prob']
	# 	return max_s_prime