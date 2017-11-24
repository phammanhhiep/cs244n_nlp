import sys, os
import pytest
from collections import defaultdict
sys.path.insert (0, os.getcwd ())
import hmm_tagger_2
HMM_Tagger_2 = hmm_tagger_2.HMM_Tagger_2

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

# @pytest.mark.skip ()
def test_smooth ():
	sents = [['this', 'is', 'a', 'dog'], ['It', 'is', 'friendly']]
	tagsq = [['pp', 'aux', 'at', 'nn'], ['pp', 'aux', 'adj']]
	O = ['It', 'is', 'a', 'friendly', 'a', 'dog']
	# tags = ['pp', 'aux', 'at', 'adj', 'at', 'nn']
	params = HMM_Tagger_2.train (sents, tagsq, ngram=3)
	params = HMM_Tagger_2.smooth (params, O, ngram=3)
	lambda_set = HMM_Tagger_2.estimate_lambda_set (params, 3)
	assert params['A']['adj']['cond']['aux at']['prob'] == params['A']['adj']['prob'] * lambda_set[0]
	assert params['A']['nn']['cond']['adj at']['prob'] == params['A']['nn']['prob'] * lambda_set[0] + params['A']['nn']['cond']['at']['prob'] * lambda_set[1]

	params = HMM_Tagger_2.train (sents, tagsq)
	params = HMM_Tagger_2.smooth (params, O)
	lambda_set = HMM_Tagger_2.estimate_lambda_set (params, 2)
	assert params['A']['at']['cond']['adj']['prob'] == params['A']['adj']['prob'] * lambda_set[0]

def test_estimate_lambda_set (): pass

def test_handle_unknown_word (): pass

@pytest.mark.skip ()
def test_gen_combined_t_primes ():
	sents = [['this', 'is', 'a', 'dog'], ['It', 'is', 'friendly'], ['It', 'was']]
	tagsq = [['pp', 'aux', 'at', 'nn'], ['pp', 'aux', 'adj'], ['subj', 'aux']]
	O = ['It', 'is', 'a', 'friendly', 'a', 'dog']
	# tags = ['pp', 'aux', 'at', 'adj', 'at', 'nn']
	params = HMM_Tagger_2.train (sents, tagsq, ngram=3)
	processed_O = HMM_Tagger_2.preprocess_obs (O, '<s>', '</s>', ngram=3)
	i = 3
	t_primes = HMM_Tagger_2.gen_combined_t_primes (params, processed_O, i, ngram=3)
	assert len (t_primes) == 2
	for tp in t_primes: assert tp in [('<s>','pp'), ('<s>','subj')]


