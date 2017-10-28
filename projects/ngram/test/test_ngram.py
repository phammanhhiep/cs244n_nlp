import os, sys
sys.path.insert (0, os.getcwd ())

from collections import defaultdict
import pytest
import ngram

def test_ngram_count_2 ():
	sent = ['<s>', 'I', 'am', 'robot', '.', '</s>']
	params = defaultdict (lambda: {
		'count': 0, 
		2: defaultdict (lambda: {'count': 0}, {}), 
		3: defaultdict (lambda: {'count': 0}, {})
	}, {})

	n = ngram.Ngram ()
	params = n.count (sent, params)

	for k,v in params.items ():
		assert v['count'] == 1
		assert len (v[2]) == 1

def test_ngram_count_3 ():
	sent = ['<s>', 'I', 'am', 'robot', '.', '</s>']
	params = defaultdict (lambda: {
		'count': 0, 
		2: defaultdict (lambda: {'count': 0}, {}), 
		3: defaultdict (lambda: {'count': 0}, {})
	}, {})

	n = ngram.Ngram ()
	params = n.count (sent, params,3)

	for k,v in params.items ():
		assert v['count'] == 1

def test_ngram_handle_unknown (): pass
