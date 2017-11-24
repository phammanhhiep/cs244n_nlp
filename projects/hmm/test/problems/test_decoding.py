import sys, os
sys.path.insert (0, os.getcwd ())

import pytest
import problems.decoding as dcd

@pytest.fixture
def L ():
	return {
		'A': { # transition p. Do not confuse the index with observation values. The indices correspond to the hidden states from start to end. This is a state * state matrix
			'<s>': {'cond': {}},
			'HOT': {'cond': {'HOT': {'prob': 0.6}, 'COLD': {'prob': 0.4}, '<s>': {'prob': 0.8}}},
			'COLD': {'cond': {'HOT': {'prob': 0.3}, 'COLD': {'prob': 0.5}, '<s>': {'prob': 0.2}}},
			'</s>': {'cond': {'HOT': {'prob': 0.1}, 'COLD': {'prob': 0.1}}},
		},
		'B': { # emission p		
			1: {'cond': {'HOT': {'prob': 0.2}, 'COLD': {'prob': 0.5}}},
			2: {'cond': {'HOT': {'prob': 0.4}, 'COLD': {'prob': 0.4}}}, 
			3: {'cond': {'HOT': {'prob': 0.4}, 'COLD': {'prob': 0.1}}},
		} 
	}

# @pytest.mark.skip ()
def test_get_max_path ():
	viterbi = {
		'<s>': [-1,-1,-1], # <s>
		'HOT': [-1,3,4], # HOT
		'COLD': [-1,4,7], # COLD
		'</s>': [-1,-1,-1], # </s>
	}

	pointers = {
		'<s>': [-1,-1,-1, -1],
		'HOT': [-1,('<s>',0),(2,1),-1],
		'COLD': [-1,(0,0),('HOT',1),-1],
		'</s>': [-1,-1,-1,('COLD',2)],
	}

	path= dcd.get_max_path (pointers, viterbi)
	pnum = len (path)
	expected_path = ['<s>', 'HOT', 'COLD']
	for i in range (pnum):
		assert path[i] == expected_path[i]

# @pytest.mark.skip ()
def test_estimate_forward_p_1 (L):
	viterbi = {
		'<s>': [-1,-1,-1,-1], # <s>
		'HOT': [-1,3,4,None], # HOT
		'COLD': [-1,4,7,None], # COLD
		'</s>': [-1,5,7,-1], # </s>
	}

	O = [3,3,1,1,2,2,3,1,3]
	s = 'COLD'
	t = 3  
	max_s_prime = dcd.estimate_forward_p (viterbi, s, t, L, O)
	assert viterbi[s][t] == 7 * L['A'][s]['cond']['COLD']['prob'] * L['B'][O[t]]['cond'][s]['prob']
	assert max_s_prime == 'COLD'

# @pytest.mark.skip ()
def test_estimate_forward_p_2 (L):
	# initialization of the viterbi
	viterbi = {
		'<s>': [-1,-1,-1,-1], # <s>
		'HOT': [-1,3,4,None], # HOT
		'COLD': [-1,4,7,None], # COLD
		'</s>': [-1,5,7,-1], # </s>
	}

	O = [3,3,1,1,2,2,3,1,3]
	s = 'HOT'
	t = 1  
	max_s_prime = dcd.estimate_forward_p (viterbi, s, t, L, O)
	assert viterbi[s][t] == L['A'][s]['cond']['<s>']['prob'] * L['B'][O[t]]['cond'][s]['prob']
	assert max_s_prime == '<s>'

# @pytest.mark.skip ()
def test_estimate_forward_p_2 (L):
	# Termination of the viterbi
	viterbi = {
		'<s>': [-1,-1,-1,-1], # <s>
		'HOT': [-1,3,4,4], # HOT
		'COLD': [-1,4,7,6], # COLD
		'</s>': [-1,5,7,-1], # </s>
	}

	O = [3,3,1]
	s = '</s>'
	t =  3 
	max_s_prime = dcd.estimate_forward_p (viterbi, s, t, L, O)
	assert viterbi[s][t] == viterbi['COLD'][t]
	assert max_s_prime == 'COLD'

# @pytest.mark.skip ()
def test_decode ():
	L = {
		'A': { # transition p. Do not confuse the index with observation values. The indices correspond to the hidden states from start to end. This is a state * state matrix
			'<s>': {'cond': {}},
			'HOT': {'cond': {'HOT': {'prob': 2}, 'COLD': {'prob': 1}, '<s>': {'prob':1}}},
			'COLD': {'cond': {'HOT': {'prob': 1}, 'COLD': {'prob': 2}, '<s>': {'prob': 2}}},
			'</s>': {'cond': {'HOT': {'prob': 1}, 'COLD': {'prob': 1}}},
		},
		'B': { # emission p		
			1: {'cond': {'HOT': {'prob': 2}, 'COLD': {'prob': 1}}},
			2: {'cond': {'HOT': {'prob': 1}, 'COLD': {'prob': 2}}}, 
			3: {'cond': {'HOT': {'prob': 2}, 'COLD': {'prob': 1}}},
		} 
	}
	O = [2,1,1,3]

	path, p = dcd.decode (L, O)
	assert p == 128
