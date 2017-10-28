import os, sys
sys.path.insert (0, os.getcwd ())

from collections import defaultdict
import pytest, math
import backoff

def test_backoff_count (): 
	m = backoff.Backoff ()
	sent = ['<s>', '<s>', 'i','am','robot', '</s>']
	ngram = 3
	def _gen_word_dict (ngram):
		w = {'count': 0, 'single': False}
		w.update ({ngram-k: defaultdict (lambda: {'count': 0, 'logp': 0}, {}) for k in range (ngram-1)})
		return w
	params = defaultdict (lambda: _gen_word_dict (ngram), {})
	params = m.count (sent, params, ngram)

	assert len (params) == 9
	assert params['<s>']['count'] == 1
	assert params['i']['count'] == 1
	assert params['robot']['count'] == 1
	assert params['i'][3]['<s> <s>']['count'] == 1
	assert params['i'][2]['<s>']['count'] == 1
	assert params['robot'][3]['i am']['count'] == 1
	assert params['robot'][2]['am']['count'] == 1	

def test_backoff_estimate_logp ():
	m = backoff.Backoff ()
	sent = ['3x', '2x', '4x', 'x']
	ngram = 3
	params = {
		'x': {
			'count': 1,
			'single': True,
			3: {
				'2x 4x': {'count': 0}
			},
			2: {
				'4x': {'count': 0.5}
			}
		},
		'2x 4x': {'count': 1, 'single': False},
		'4x': {
			'count': 1,
			'single': True,
			3: {
				'3x 2x': {'count': 1}
			},
			2: {
				'2x': {'count': 1}
			}
		},
		'3x 2x': {'count': 5, 'single': False}
	}

	params = m.estimate_logp (params, ngram)
	assert params['x'][ngram]['2x 4x']['logp'] == math.log (0.5)
	assert params['4x'][ngram]['3x 2x']['logp'] == math.log (1/5)

	def test_backoff_evaluate ():
		