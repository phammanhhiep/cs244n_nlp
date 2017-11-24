import sys, os
sys.path.insert (0, os.getcwd ())

import pytest
import decoder
Decoder = decoder.Decoder

@pytest.fixture
def L ():
	return {
		'A': { # transition p. Do not confuse the index with observation values. The indices correspond to the hidden states from start to end. This is a state * state matrix
			'<s>': {'cond': {}},
			'HOT': {'cond': {'HOT': {'prob': 0.6}, 'COLD': {'prob': 0.4}, '<s>': {'prob': 0.8}, '<s> <s>': {'prob': 0.4}, 'COLD COLD': {'prob': 0.7}, 'COLD HOT': {'prob': 0.6},'HOT COLD': {'prob': 0.4},'HOT HOT': {'prob': 0.3}, '<s> HOT': {'prob': 0.7}, '<s> COLD': {'prob': 0.7}}},
			'COLD': {'cond': {'HOT': {'prob': 0.3}, 'COLD': {'prob': 0.5}, '<s>': {'prob': 0.2}, '<s> <s>': {'prob': 0.6}, 'COLD COLD': {'prob': 0.3}, 'COLD HOT': {'prob': 0.4},'HOT COLD': {'prob': 0.6},'HOT HOT': {'prob': 0.7}, '<s> HOT': {'prob': 0.7}, '<s> COLD': {'prob': 0.7}}},
			'</s>': {'cond': {'HOT': {'prob': 0.1}, 'COLD': {'prob': 0.1}, 'COLD HOT': {'prob': 0.1},'HOT COLD': {'prob': 0.1}, 'HOT HOT': {'prob': 0.1}, 'COLD COLD': {'prob': 0.1}}},
		},
		'B': { # emission p	
			'<s>': {'cond': {'<s>': {'prob': 1}}},	
			1: {'cond': {'HOT': {'prob': 0.2}, 'COLD': {'prob': 0.5}}},
			2: {'cond': {'HOT': {'prob': 0.4}, 'COLD': {'prob': 0.4}}}, 
			3: {'cond': {'HOT': {'prob': 0.4}, 'COLD': {'prob': 0.1}}},
			'</s>': {'cond': {'</s>': {'prob': 1}}},

		} 
	}

@pytest.mark.skip ()
def test_get_max_path ():
	pointers = {
		'<s>': [-1,-1,-1,-1,-1],
		'HOT': [-1,'<s>',-1,-1,-1],
		'COLD': [-1,-1,'HOT','COLD',-1],
		'</s>': [-1,-1,-1,-1,'COLD'],
	}

	path= Decoder.get_max_path (pointers)
	pnum = len (path)
	expected_path = ['<s>', 'HOT', 'COLD', 'COLD']
	for i in range (pnum):
		assert path[i] == expected_path[i]

@pytest.mark.skip ()
def test_get_prev_states ():
	# pointers = {
	# 	'<s>': [-1,-1,-1,-1,-1],
	# 	'HOT': [-1,'<s>','COLD',-1,-1],
	# 	'COLD': [-1,'<s>','HOT',1,-1],
	# 	'</s>': [-1,-1,-1,-1,-1],
	# }

	# t=3; ngram=2
	# states = Decoder.get_prev_states (pointers, t, ngram)
	# assert len (states) == 2
	# for s in states: assert s in ['COLD', 'HOT']

	pointers = {
		'<s>': [-1,-1,-1,-1,-1,-1],
		'HOT': [-1,-1,'<s>','COLD','COLD',-1],
		'COLD': [-1,-1,'<s>','HOT','HOT',-1],
		'</s>': [-1,-1,-1,-1,-1,-1],
	}

	t=4; ngram=3
	states = Decoder.get_prev_states (pointers, t, ngram)
	assert len (states) == 2
	for s in states: assert s in ['HOT COLD', 'COLD HOT']

	t=5; ngram=4
	states = Decoder.get_prev_states (pointers, t, ngram)
	assert len (states) == 2
	for s in states: assert s in ['HOT COLD HOT', 'COLD HOT COLD']	

@pytest.mark.skip ()
def test_estimate_forward_p_1 (L):
	# initialization of the viterbi
	viterbi = {
		'<s>': [-1,-1,-1,-1], # <s>
		'HOT': [-1,3,4,None], # HOT
		'COLD': [-1,4,7,None], # COLD
		'</s>': [-1,5,7,-1], # </s>
	}

	pointers = {}
	O = ['<s>',3,3,1,1,2,2,3,1,3,'</s>']
	s = 'HOT'
	t = 1  
	ngram = 2
	max_s_prime = Decoder.estimate_forward_p (viterbi, pointers, s, t, L, O, ngram=ngram)
	assert viterbi[s][t] == L['A'][s]['cond']['<s>']['prob'] * L['B'][O[t]]['cond'][s]['prob']
	assert max_s_prime == '<s>'

	viterbi = {
		'<s>': [-1,-1,-1,-1,-1], # <s>
		'HOT': [-1,-1,3,4,None], # HOT
		'COLD': [-1,-1,4,7,None], # COLD
		'</s>': [-1,-1,5,7,-1], # </s>
	}

	t = 2 
	ngram = 3
	max_s_prime = Decoder.estimate_forward_p (viterbi, pointers, s, t, L, O, ngram=ngram)
	assert viterbi[s][t] ==  L['A'][s]['cond']['<s> <s>']['prob'] * L['B'][O[t]]['cond'][s]['prob']
	assert max_s_prime == '<s> <s>'

@pytest.mark.skip ()
def test_estimate_forward_p_2 (L):
	# iterate the viterbi
	viterbi = {
		'<s>': [-1,-1,-1,-1], # <s>
		'HOT': [-1,3,4,None], # HOT
		'COLD': [-1,4,7,None], # COLD
		'</s>': [-1,-1,-1,-1], # </s>
	}

	pointers = {
		'<s>': [-1,-1,-1,-1,-1],
		'HOT': [-1,'<s>','COLD',-1,-1],
		'COLD': [-1,'<s>','HOT',1,-1],
		'</s>': [-1,-1,-1,-1,-1],
	}	

	O = ['<s>',3,3,1,1,2,2,3,1,3,'</s>']
	s = 'COLD'
	t = 3 
	ngram=2
	max_s_prime = Decoder.estimate_forward_p (viterbi, pointers, s, t, L, O, ngram=ngram)
	assert viterbi[s][t] == 7 * L['A'][s]['cond']['COLD']['prob'] * L['B'][O[t]]['cond'][s]['prob']
	assert max_s_prime == 'COLD'

	viterbi = {
		'<s>': [-1,-1,-1,-1,-1,-1],
		'HOT': [-1,-1,3,4,5,-1],
		'COLD': [-1,-1,4,7,3,-1],
		'</s>': [-1,-1,-1,-1,-1,-1],
	}

	pointers = {
		'<s>': [-1,-1,-1,-1,-1,-1],
		'HOT': [-1,-1,'<s>','COLD','HOT',-1],
		'COLD': [-1,-1,'<s>','COLD','COLD',-1],
		'</s>': [-1,-1,-1,-1,-1,-1],
	}

	t = 5
	ngram = 3
	max_s_prime = Decoder.estimate_forward_p (viterbi, pointers, s, t, L, O, ngram=ngram)
	assert max_s_prime == 'HOT HOT'
	assert viterbi[s][t] == viterbi['HOT'][t-1] * L['A'][s]['cond'][max_s_prime]['prob'] * L['B'][O[t]]['cond'][s]['prob']
	
@pytest.mark.skip ()
def test_estimate_forward_p_3 (L):
	# Termination of the viterbi
	viterbi = {
		'<s>': [-1,-1,-1,-1,-1], # <s>
		'HOT': [-1,3,4,4,-1], # HOT
		'COLD': [-1,4,7,6,-1], # COLD
		'</s>': [-1,-1,-1,-1,-1], # </s>
	}

	pointers = {
		'<s>': [-1,-1,-1,-1,-1],
		'HOT': [-1,-1,'<s>','COLD',-1],
		'COLD': [-1,-1,'<s>','HOT',-1],
		'</s>': [-1,-1,-1,-1,-1],
	}

	O = ['<s>',3,3,1,'</s>']
	s = '</s>'
	t =  4 
	ngram = 2
	max_s_prime = Decoder.estimate_forward_p (viterbi, pointers, s, t, L, O, ngram=ngram)
	assert max_s_prime == 'COLD'
	assert viterbi[s][t] == viterbi['COLD'][t-1] * L['A'][s]['cond']['COLD']['prob']
	
	ngram = 3
	max_s_prime = Decoder.estimate_forward_p (viterbi, pointers, s, t, L, O, ngram=ngram)
	assert max_s_prime == 'HOT COLD'
	assert viterbi[s][t] == 6 * L['A'][s]['cond'][max_s_prime]['prob']
		
# @pytest.mark.skip ()
def test_decode (L):
	# TEST later.
	# L = {
	# 	'A': { # transition p. Do not confuse the index with observation values. The indices correspond to the hidden states from start to end. This is a state * state matrix
	# 		'<s>': {'cond': {}},
	# 		'HOT': {'cond': {'HOT': {'prob': 2}, 'COLD': {'prob': 1}, '<s>': {'prob':1}}},
	# 		'COLD': {'cond': {'HOT': {'prob': 1}, 'COLD': {'prob': 2}, '<s>': {'prob': 2}}},
	# 		'</s>': {'cond': {'HOT': {'prob': 1}, 'COLD': {'prob': 1}}},
	# 	},
	# 	'B': { # emission p
	# 		'<s>': {'cond': {'<s>': {'prob': 1}}},
	# 		1: {'cond': {'HOT': {'prob': 2}, 'COLD': {'prob': 1}}},
	# 		2: {'cond': {'HOT': {'prob': 1}, 'COLD': {'prob': 2}}}, 
	# 		3: {'cond': {'HOT': {'prob': 2}, 'COLD': {'prob': 1}}},
	# 		'</s>': {'cond': {'</s>': {'prob': 1}}},
	# 	} 
	# }

	O = ['<s>',2,1,1,3,'</s>']
	ngram = 2
	path, p = Decoder.decode (L, O, ngram=ngram)
	print (path, p)
	# assert p == 128
	# pnum = len (path)
	# expected_path = []
	# for i in range (pnum): path[i] == expected_path[i]

	O = ['<s>','<s>', 2,1,1,3,'</s>']
	ngram = 3
	path, p = Decoder.decode (L, O, ngram=ngram)
	print (path, p)
	# assert p == 128
	# pnum = len (path)
	# expected_path = []
	# for i in range (pnum): path[i] == expected_path[i]

