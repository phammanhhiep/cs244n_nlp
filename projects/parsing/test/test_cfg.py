import os, sys
sys.path.insert (0, os.getcwd ())
import cfg

# from collections import defaultdict
# import random, math 
import pytest

@pytest.fixture
def CFG ():
	cfg = {
		'rules': {
			'S': ['NP VP', 'Aux NP VP', 'VP'],
			'NP': ['Pronoun', 'Proper-Noun', 'Det Nominal'],
			'Nominal': ['Noun', 'Nominal Noun', 'Nominal PP'],
			'VP': ['Verb', 'Verb NP', 'Verb NP PP', 'Verb PP'],
			'PP': ['Preposition PP']
		},
		'lexicon': {
			'Det': ['that', 'this', 'the', 'a'],
			'Noun': ['book', 'flight', 'meal', 'money'],
			'Verb': ['book', 'include', 'prefer'],
			'Pronoun': ['I', 'she', 'me'],
			'Proper-Noun': ['Houston', 'NWA'],
			'Aux': ['does'],
			'Preposition': ['from', 'to', 'on', 'near', 'through']
		}
	}

	return cfg

def test__has_unit_production ():
	CFG = { # arbitrary
		'rules': {
			'S': ['NP PP', 'NP', 'Aux NP PP'],
			'VP': ['Verb Adverb', 'Verb PP']
		}
	}

	has_up = cfg._has_unit_production ('S', CFG['rules'])
	assert has_up is True
	has_up = cfg._has_unit_production ('VP', CFG['rules'])
	assert has_up is False	

def test__resolve_unit_production_1 ():
	# when the right-side non-terminal of UP derives a terminal
	CFG = { # arbitrary
		'rules': {
			'S': ['Verb'],
			'PP': ['Verb']
		},
		'lexicon': {
			'Verb': ['x', 'y'],
			'PP': ['z', 'y']
		}
	}

	CNF = {
		'rules': {},
		'lexicon': CFG['lexicon']
	}

	cfg._resolve_unit_production ('Verb', 'S', CFG['rules'], CNF)
	cfg._resolve_unit_production ('Verb', 'PP', CFG['rules'], CNF)

	assert len (CNF['lexicon']['S']) == 2
	for i in ['x', 'y']:
		assert i in CNF['lexicon']['S']
	assert len (CNF['lexicon']['PP']) == 3
	for i in ['x', 'y', 'z']:
		assert i in CNF['lexicon']['PP']	

def test__resolve_unit_production_2 ():
	# when the right-side non-terminal of UP derives which has no unit production
	CFG = { # arbitrary
		'rules': {
			'S': ['VP', 'VP PP'],
			'VP': ['Verb PP', 'Verb NP'],
			'K': ['Z', 'VP PP'], # arbitrary
			'Z': ['Verb PP', 'VP PP'], # arbitrary
		},
		'lexicon': {
			'Verb': ['x', 'y'],
			'PP': ['z', 'y']
		}
	}

	CNF = {
		'rules': {},
		'lexicon': CFG['lexicon']
	}

	cfg._resolve_unit_production ('VP', 'S', CFG['rules'], CNF)
	cfg._resolve_unit_production ('Z', 'K', CFG['rules'], CNF)
	assert len (CFG['rules']['S']) == 4
	assert None in CFG['rules']['S']
	for t in ['Verb PP', 'Verb NP']: assert t in CFG['rules']['S']

	assert len (CFG['rules']['K']) == 3
	assert None in CFG['rules']['K']
	for t in ['Verb PP', 'VP PP']: assert t in CFG['rules']['K']

def test__resolve_unit_production_3 ():
	# when the right-side non-terminal of UP derives one unit production 
	CFG = { # arbitrary
		'rules': {
			'S': ['VP', 'VP PP'],
			'VP': ['K', 'Verb PP', 'Verb NP'],
			'K': ['Z T', 'VP PP'], # arbitrary
			'Z': ['Verb PP', 'VP PP'], # arbitrary
		},
		'lexicon': {
			'Verb': ['x', 'y'],
			'PP': ['z', 'y']
		}
	}

	CNF = {
		'rules': {},
		'lexicon': CFG['lexicon']
	}

	cfg._resolve_unit_production ('VP', 'S', CFG['rules'], CNF)
	assert len (CFG['rules']['S']) == 5
	assert None in CFG['rules']['S']
	for t in ['Verb PP', 'Verb NP', 'Z T', 'VP PP']: assert t in CFG['rules']['S'] 

def test__resolve_unit_production_4 ():
	# when UP derives non-terminal, which has more than one unit production
	CFG = { # arbitrary
		'rules': {
			'S': ['VP', 'VP PP'],
			'VP': ['K', 'Verb PP', 'T'],
			'K': ['Z T', 'VP PP'], # arbitrary
			'T': ['Adverb Verb PP', 'VP PP'], # arbitrary
		},
		'lexicon': {
			'Verb': ['x', 'y'],
			'PP': ['z', 'y']
		}
	}

	CNF = {
		'rules': {},
		'lexicon': CFG['lexicon']
	}

	cfg._resolve_unit_production ('VP', 'S', CFG['rules'], CNF)
	assert len (CFG['rules']['S']) == 5
	assert None in CFG['rules']['S']
	for t in ['Verb PP', 'Adverb Verb PP', 'Z T', 'VP PP']: assert t in CFG['rules']['S']
	for t in ['Verb PP', 'Adverb Verb PP', 'Z T', 'VP PP']: assert t in CFG['rules']['VP']	

def test__resolve_more2_nonterminal_ ():
	# there are three or more nonterminals
	CFG = { # arbitrary
		'rules': {
			'S': ['VP', 'VP PP XX'],
			'NP': ['XX YY ZZ TT', 'NM IQ']
		}
	}

	d = 0
	d = cfg._resolve_more2_nonterminal (d, 'VP PP XX', 1, 'S', CFG['rules'])
	d = cfg._resolve_more2_nonterminal (d, 'XX YY ZZ TT', 0, 'NP', CFG['rules'])

	assert d == 3
	assert CFG['rules']['S'][1] == 'Dummy_1 XX'
	assert CFG['rules']['Dummy_1'][0] == 'VP PP'
	assert CFG['rules']['NP'][0] == 'Dummy_2 TT'
	assert CFG['rules']['Dummy_2'][0] == 'Dummy_3 ZZ'
	assert CFG['rules']['Dummy_3'][0] == 'XX YY'

def test__is_mixed ():
	ri1 = 'XX walk YY'
	ri2 = 'RUNN'
	L = {
		'ADJ': ['lazy'],
		'Verb': ['run', 'walk']
	}

	assert cfg._is_mixed (ri1, L) is True
	assert cfg._is_mixed (ri2, L) is False

def test__resolve_mixed ():
	CFG = { # arbitrary
		'rules': {
			'S': ['VP', 'VP PP', 'Verb yy zz']
		},
		'lexicon': {
			'Verb': ['x', 'yy'],
			'PP': ['zz', 'y']
		}
	}

	dcount = 0
	dcount = cfg._resolve_mixed (dcount, 'Verb yy zz', 2, 'S', CFG['rules'], CFG['lexicon'])
	assert dcount == 2
	CFG['lexicon']['Dummy_1'][0] == 'yy'
	CFG['lexicon']['Dummy_2'][0] == 'zz'
	CFG['rules']['S'][2] = 'Verb Dummy_1 Dummy_2'

def test__clean_none ():
	G = {
		'X': ['zz', None, 'yy'],
		'Y': [None, None],
		'Z': ['xx', 'aa']
	}

	cfg._clean_none (G['X'], 'X', G)
	cfg._clean_none (G['Y'], 'Y', G)
	cfg._clean_none (G['Z'], 'Z', G)

	assert len (G['X']) == 2
	assert None not in G['X']
	assert len (G['Y']) == 0	  
	assert len (G['Z']) == 2
	assert None not in G['Z']

def test_to_CNF (CFG):
	CNF = cfg.to_CNF (CFG)
	assert len (CNF['rules']['S']) == 5
	assert len (CNF['lexicon']['S']) == 3


	
