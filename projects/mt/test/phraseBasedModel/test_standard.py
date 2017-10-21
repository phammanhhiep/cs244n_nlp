import os, sys
sys.path.insert (0, os.getcwd ())

from collections import defaultdict
import pytest
import phraseBasedModels.standard as standard

def test_standard__extract_phrase ():
	tsent = ['xxx', 'xx', 'xxxx', 'x']
	ssent = ['y', 'yyy', 'yy', 'yyyy']
	fj_start, fj_end, ei_start, ei_end = 3,4,2,3
	sd = standard.Standard () 
	wa = [(1,2), (2,3), (3,4), (4,1)]
	ephrase, fphrases = sd._extract_phrase (fj_start, fj_end, ei_start, ei_end, ssent, tsent, wa)

	assert ' '.join (ephrase) == 'xx xxxx'
	assert len (fphrases) == 1
	assert ' '.join (fphrases[0]) == 'yy yyyy'

def test_standard__is_aligned_index ():
	wa = [(1,2), (3,5), (2,4)]
	fj = 2
	sd = standard.Standard () 
	is_aligned = sd._is_aligned_index (fj, wa)
	assert is_aligned is True


def test_standard__extract_phrases_by_sent ():
	# assume to be True
	tsent = ['xxx', 'xx', 'xxxx', 'x']
	ssent = ['y', 'yyy', 'yy', 'yyyy', 'z']
	sd = standard.Standard () 
	wa = [(1,2), (2,3), (3,4), (4,1)]
	phrases = defaultdict (lambda: [], {})
	phrases = sd._extract_phrases_by_sent (phrases, ssent, tsent, wa)

	assert True 


def test_standard_count_phrases (): pass

def test_estimate_phrase_probability (): pass