import sys, os
sys.path.insert (0, os.getcwd ())

import pytest
import problems.decoding as dcd

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

def test_get_max_viterbi_1 ():
	# in initialization
	viterbi = [
		[-1,3,4],
		[-1,4,7]
	] 
	t = 0
	max_p, max_s_prime = dcd.get_max_viterbi (viterbi, t)
	assert max_p == 1
	assert max_s_prime == 0

def test_get_max_viterbi_2 ():
	# in other phases
	viterbi = [
		[-1,3,4],
		[-1,4,7]
	] 
	t = 1
	max_p, max_s_prime = dcd.get_max_viterbi (viterbi, t)
	assert max_p is 4
	assert max_s_prime is 1

def test_get_max_path ():
	viterbi = [
		[-1,-1,-1],
		[-1,3,4,],
		[-1,4,7,],
	] 

	pointers = [
		[-1,-1,-1,(2,2)],
		[-1,(0,0),(2,1),-1],
		[-1,(0,0),(1,1),-1]
	]

	path, p = dcd.get_max_path (pointers, viterbi)
	assert p == 21
	for i in path:
		assert i in [0,1,2]

# @pytest.mark.skip ()
def test_decode (L):
	O1 = [3,3,1,1,2,2,3,1,3]
	O2 = [3,3,1,1,2,3,3,1,2]

	path1, p1 = dcd.decode (L, O1)
	path2, p2 = dcd.decode (L, O2)

	print (path1, p1)
	print (path2, p2)

	assert True
