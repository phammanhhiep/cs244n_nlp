import os, sys
sys.path.insert (0, os.getcwd ())

from collections import defaultdict

import pytest
import wordAlignment


def test_pmi ():
	assert True is True

def test_IBM_model_1_get_delta (): pass

def test_IBM_model_1_get_q (): pass

def test_IBM_model_1_get_t (): pass

def test_em_model_1 ():
	scorpus = [['xxx', 'x', 'xxxx', 'xx']]
	tcorpus = [['this', 'is', 'my', 'house']]
	expected_qj = expected_tj = 0.25

	get_q = wordAlignment.IBMModel1.get_q
	get_t= wordAlignment.IBMModel1.get_t
	get_delta = wordAlignment.IBMModel1.get_delta

	em = wordAlignment.EM ()
	q,t = em.run (scorpus, tcorpus, get_q, get_t, get_delta, 5)

	for k,v in q.items ():
		assert v == expected_qj

	for k,v in t.items ():
		for kj,vj in v.items ():
			assert vj == expected_tj		


def test_em_model_2 ():
	scorpus = [['xxx', 'x', 'xxxx', 'xx']]
	tcorpus = [['this', 'is', 'my', 'house']]
	expected_qj = expected_tj = 0.25

	get_q = wordAlignment.IBMModel2.get_q
	get_t= wordAlignment.IBMModel2.get_t
	get_delta = wordAlignment.IBMModel2.get_delta

	em = wordAlignment.EM ()
	q,t = em.run (scorpus, tcorpus, get_q, get_t, get_delta, 5)

	for k,v in q.items ():
		assert v == expected_qj

	for k,v in t.items ():
		for kj,vj in v.items ():
			assert vj == expected_tj