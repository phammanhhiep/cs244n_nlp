import sys, os
sys.path.insert (0, os.getcwd ())

import pytest
import problems.likelihood as ll

@pytest.fixture
def L ():
	return {
		'A': { # transition p. Do not confuse the index with observation values. The indices correspond to the hidden states from start to end. This is a state * state matrix
			0: [0, 0.8, 0.2, 0],
			1: [0, 0.6, 0.3, 0.1], # HOT
			2: [0, 0.4, 0.5, 0.1], # COLD
			3: [0, 0, 0, 0]
		},
		'B': { # emission p		
			1: [0, 0.2, 0.4, 0.4], # HOT
			2: [0, 0.5, 0.4, 0.1], # COLD
		} 
	}

def test_sum_forward (L):
	O = [3,3,1,1,2,2,3,1,3]
	forward = [
		[None] * (len (O) + 1),
		[None, L['A'][0][1] * L['B'][1][O[0]], None],
		[None, L['A'][0][2] * L['B'][2][O[0]], None],
		[None] * (len (O) + 1)
	]

	t = 2
	s = 1
	total = ll.sum_forward (forward, L, O, t, s)
	assert total == L['A'][0][1] * L['B'][1][O[0]] * L['A'][1][s] * L['B'][s][O[t]] + L['A'][0][2] * L['B'][2][O[0]] * L['A'][2][s] * L['B'][s][O[t]]

# @pytest.mark.skip ()
def test_estimate_likelihood (L):
	O1 = [3,3,1,1,2,2,3,1,3]
	O2 = [3,3,1,1,2,3,3,1,2]
	p1, forward1 = ll.estimate_likelihood (L, O1)
	p2, forward1 = ll.estimate_likelihood (L, O2)
	print (p1, p2)
	assert True	