import sys, os
import pytest
from collections import defaultdict
sys.path.insert (0, os.getcwd ())
import hmm_tagger_2
HMM_Tagger_2 = hmm_tagger_2.HMM_Tagger_2
UNK_Handler = hmm_tagger_2.UNK_Handler

@pytest.mark.skip ()
def test_train_count ():
	sents = [['this', 'is', 'a', 'dog'], ['It', 'is', 'friendly']]
	tagsq = [['pp', 'aux', 'at', 'nn'], ['pp', 'aux', 'adj']]
	params = HMM_Tagger_2.gen_params ()

	params = HMM_Tagger_2._count (sents, tagsq, params)

	assert params['B']['is']['count'] == 2
	assert params['B']['is']['cond']['aux']['count'] == 2
	assert params['A']['pp']['cond']['<s>']['count'] == 2
	assert params['A']['aux']['cond']['pp']['count'] == 2
	assert params['A']['at']['cond']['aux']['count'] == 1
	assert params['A']['</s>']['cond']['nn']['count'] == 1
	assert params['A']['</s>']['count'] == 2
	assert params['A']['<s>']['count'] == 2

@pytest.mark.skip ()
def test_smooth_transition ():
	sents = [['this', 'is', 'a', 'dog'], ['It', 'is', 'friendly']]
	tagsq = [['pp', 'aux', 'at', 'nn'], ['pp', 'aux', 'adj']]
	O = ['It', 'is', 'a', 'friendly', 'a', 'dog']
	# tags = ['pp', 'aux', 'at', 'adj', 'at', 'nn']
	params = HMM_Tagger_2.train (sents, tagsq, ngram=3)
	params = HMM_Tagger_2.smooth_transition (params, O, ngram=3)
	lambda_set = HMM_Tagger_2.estimate_lambda_set (params, 3)
	assert params['A']['adj']['cond']['aux at']['prob'] == params['A']['adj']['prob'] * lambda_set[0]
	assert params['A']['nn']['cond']['adj at']['prob'] == params['A']['nn']['prob'] * lambda_set[0] + params['A']['nn']['cond']['at']['prob'] * lambda_set[1]

	params = HMM_Tagger_2.train (sents, tagsq)
	params = HMM_Tagger_2.smooth_transition (params, O)
	lambda_set = HMM_Tagger_2.estimate_lambda_set (params, 2)
	assert params['A']['at']['cond']['adj']['prob'] == params['A']['adj']['prob'] * lambda_set[0]

def test_estimate_lambda_set ():
	params = {
		'A': {
			'nn': {'count': 5, 'cond': {'verb adj': {'count': 4}, 'adj': {'count': 2}}},
			'adj': {'count': 10, 'cond': {'verb': {'count': 7}, 'ng verb': {'count': 0}}},
			'verb': {'count': 7, 'cond': {'ng': {'count': 1}}},
			'ng': {'count': 2, 'cond': {}},
		},
		'_stat_':{'token': {'count': 100}}
	}

	ngram = 2
	lambda_set = HMM_Tagger_2.estimate_lambda_set (params, ngram)
	expected_ls = [0, 1]
	n = len (expected_ls)
	for i in range (n): expected_ls[i] == lambda_set[i]

	ngram = 3
	lambda_set = HMM_Tagger_2.estimate_lambda_set (params, ngram)
	expected_ls = [0, 7/11, 4/11]
	n = len (expected_ls)
	for i in range (n): expected_ls[i] == lambda_set[i]

def test_handle_unknown_word (): pass

@pytest.mark.skip ()
def test_gen_t_primes ():
	sents = [['this', 'is', 'a', 'dog'], ['It', 'is', 'friendly'], ['It', 'was']]
	tagsq = [['pp', 'aux', 'at', 'nn'], ['pp', 'aux', 'adj'], ['subj', 'aux']]
	O = ['It', 'is', 'a', 'friendly', 'a', 'dog']
	# tags = ['pp', 'aux', 'at', 'adj', 'at', 'nn']
	params = HMM_Tagger_2.train (sents, tagsq, ngram=3)
	processed_O = HMM_Tagger_2.preprocess_obs (O, '<s>', '</s>', ngram=3)
	i = 3
	t_primes = HMM_Tagger_2.gen_t_primes (params, processed_O, i, ngram=3)
	assert len (t_primes) == 2
	for tp in t_primes: assert tp in [('<s>','pp'), ('<s>','subj')]

@pytest.mark.skip()
def test_UNK_Handler_count ():
	params = {
		'B': {
			'house': {'cond': {'nn': {'count': 3}, 'verb': {'count': 5}}, 'count': 10},
			'advertising': {'cond': {'nn': {'count': 2}, 'verbg': {'count': 10}}, 'count': 5}
		}
	}

	max_length = 3
	startsymbol = '<s>'; endsymbol = '</s>'
	sp = UNK_Handler.count (params, max_length, startsymbol, endsymbol)

	assert sp['_stat_']['T']['count'] == 20
	assert sp['_stat_']['S']['count'] == 45
	assert sp['T']['nn']['count'] == 5
	assert sp['S']['e']['count'] == 10
	assert sp['S']['se']['count'] == 10
	assert sp['S']['ng']['cond']['nn']['count'] == 2
	assert sp['S']['ouse']['count'] == 0

@pytest.mark.skip()
def test_UNK_Handler_train ():
	params = {
		'B': {
			'house': {'cond': {'nn': {'count': 3}, 'verb': {'count': 5}}, 'count': 10},
			'advertising': {'cond': {'nn': {'count': 2}, 'verb': {'count': 10}}, 'count': 5}
		},

	}

	max_length = 3
	startsymbol = '<s>'; endsymbol = '</s>'
	sp = UNK_Handler.count (params, max_length, startsymbol, endsymbol)
	sp = UNK_Handler.train (sp)	

	assert sp['T']['nn']['prob'] == sp['T']['nn']['count'] / sp['_stat_']['T']['count']
	assert sp['S']['e']['prob'] == sp['S']['e']['count'] / sp['_stat_']['S']['count']
	assert sp['S']['ng']['cond']['nn']['prob'] == sp['S']['ng']['cond']['nn']['count'] / sp['S']['ng']['count']

@pytest.mark.skip()
def test_UNK_Handler_estimate_theta (): 
	sp = {
		'T': {
			'verb': {'prob': 0.1}, 'nn': {'prob': 0.4}
		}
	}	

	theta = UNK_Handler.estimate_theta (sp)
	assert theta == (pow(0.1 - 0.25, 2) + pow(0.4-0.25, 2)) / 1

@pytest.mark.skip()
def test_UNK_Handler_smooth ():
	sp = {
		'S': {
			'use': {
				'cond':{'nn': {'prob': 0.1}, 'verb': {'prob': 0.2}} 
			},
			'se': {
				'cond':{'nn': {'prob': 0.3}, 'verb': {'prob': 0.3}} 
			},
			'e': {
				'cond':{'nn': {'prob': 0.4}, 'verb': {'prob': 0.6}} 
			},							
		},
		'T': {
			'verb': {'prob': 0.1}, 'nn': {'prob': 0.4}
		}
	}

	s = 'use'
	theta = UNK_Handler.estimate_theta (sp)
	sp = UNK_Handler.smooth (s, theta, sp)
	expect_nn = (0.1 + theta * (0.3 + theta * (0.4 + theta * 0.4)/(1+theta))/(1+theta))/(1+theta)
	assert sp['S']['use']['cond']['nn']['prob'] == expect_nn

@pytest.mark.skip()
def test_UNK_Handler_handle_unknown_words ():
	params = {
		'B': defaultdict (lambda: {'cond': defaultdict (lambda: {'count': 0, 'prob': 0}, {}), 'count': 0, 'prob': 0}, {}),
	}
	params['B']['house'] = {'cond': {'nn': {'count': 3}, 'verb': {'count': 5}}, 'count': 10}
	params['B']['advertising'] = {'cond': {'nn': {'count': 2}, 'verb': {'count': 10}}, 'count': 5}

	O = ['sing', 'huse']
	params = UNK_Handler.handle_unknown_words (params, O, max_length=10)
	print (params)	