import sys, os
import pytest
from collections import defaultdict
sys.path.insert (0, os.getcwd ())
import hmm_tagger
HMM_Tagger = hmm_tagger.HMM_Tagger

def test_preprocess (): pass

def test_train (): pass

def test_train_count ():
	sents = [['this', 'is', 'a', 'dog'], ['It', 'is', 'friendly']]
	tagsq = [['pp', 'aux', 'at', 'nn'], ['pp', 'aux', 'adj']]
	params = {
		'A': defaultdict (lambda: {'cond': defaultdict (lambda: {'count': 0, 'prob': 1/100000}, {}), 'count': 0}, {}),
		'B': defaultdict (lambda: {'cond': defaultdict (lambda: {'count': 0, 'prob': 1/100000}, {}), 'count': 0}, {})
	}	
	params = HMM_Tagger._count (sents, tagsq, params)

	assert params['B']['is']['count'] == 2
	assert params['B']['is']['cond']['aux']['count'] == 2
	assert params['A']['pp']['cond']['<s>']['count'] == 2
	assert params['A']['<s>']['count'] == 2

def test_train_estimate_prob ():
	params = {
		'A': {
			'pp': {'cond': {'xx': {'count': 2}}, 'count': 0},
			'xx': {'count': 10, 'cond': {'<s>': {'count': 2}}},
			'<s>': {'count': 20, 'cond': {}}
		},
		'B': {
			'is': {'cond': {'xx': {'count': 5}}}
		}
	}

	params = HMM_Tagger._estimate_prob (params)
	assert params['A']['pp']['cond']['xx']['prob'] == 1/5
	assert params['A']['xx']['cond']['<s>']['prob'] == 1/10
	assert params['B']['is']['cond']['xx']['prob'] == 1/2

def test_train_estimate_prob_A (): pass

def test_train_estimate_prob_B (): pass

def test_decode (): pass

def test_evaluate (): pass