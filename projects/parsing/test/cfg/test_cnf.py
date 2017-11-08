import os, sys
sys.path.insert (0, os.getcwd ())
import cfg.cnf

from collections import defaultdict
import random, math 
import pytest

CNF = cfg.cnf.CNF

@pytest.fixture
def cfg ():
	cfg = {
		'rules': {
			'S': ['NP VP', 'Aux NP VP', 'VP'],
			'NP': ['Pronoun', 'Proper-Noun', 'Det Nominal'],
			'Nominal': ['Noun', 'Nominal Noun', 'Nominal PP'],
			'VP': ['Verb', 'Verb NP', 'Verb NP PP', 'Verb PP'],
			'PP': ['Preposition NP']
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
	cfg = { # arbitrary
		'rules': {
			'S': ['NP PP', 'NP', 'Aux NP PP'],
			'VP': ['Verb Adverb', 'Verb PP']
		}
	}

	has_up = CNF._has_unit_production ('S', cfg['rules'])
	assert has_up is True
	has_up = CNF._has_unit_production ('VP', cfg['rules'])
	assert has_up is False	

# @pytest.mark.skip ()
def test__resolve_unit_production_1 ():
	# when the right-side non-terminal of UP derives a terminal
	cfg = { # arbitrary
		'rules': {
			'S': ['Verb'],
			'PP': ['K', 'AB XY'],
			'K': ['Z', 'AC XY']
		},
		'lexicon': {
			'Verb': ['x', 'y'],
			'PP': ['z', 'y'],
			'Z': ['a', 'b']
		}
	}

	cnf = {
		'rules': defaultdict (lambda: [], {}),
		'lexicon': cfg['lexicon']
	}

	CNF._resolve_unit_production ('Verb', 'S', cfg['rules'], cnf)
	CNF._resolve_unit_production ('K', 'PP', cfg['rules'], cnf)

	assert len (cnf['lexicon']['S']) == 2
	for i in ['x', 'y']: assert i in cnf['lexicon']['S']
	assert len (cnf['rules']['S']) == 0

	assert len (cnf['lexicon']['PP']) == 4
	for i in ['y', 'z', 'a', 'b']: assert i in cnf['lexicon']['PP']
	assert len (cnf['rules']['PP']) == 2
	for i in ['AB XY', 'AC XY']: assert i in cnf['rules']['PP']

# @pytest.mark.skip ()
def test__resolve_unit_production_1_1 ():
	# when the right-side non-terminal of UP derives a terminal. Special case: all non-terminal in right side derive terminals
	cfg = { # arbitrary
		'rules': {
			'PP': ['K', 'AB XY'],
			'K': ['Z', 'T'],
		},
		'lexicon': {
			'PP': ['z', 'y'],
			'Z': ['a', 'b'],
			'T': ['n', 'm'],
			'X': ['y', 'x'],
			'Y': ['t', 'k'],
		}
	}

	cnf = {
		'rules': defaultdict (lambda: [], {}),
		'lexicon': cfg['lexicon']
	}

	CNF._resolve_unit_production ('K', 'PP', cfg['rules'], cnf)

	assert len (cnf['lexicon']['PP']) == 6
	for i in ['y', 'z', 'a', 'b', 'n', 'm']: assert i in cnf['lexicon']['PP']
	assert len (cnf['rules']['PP']) == 1
	for i in ['AB XY']: assert i in cnf['rules']['PP']

def test__resolve_unit_production_2 ():
	# when the right-side non-terminal of UP derives which has no unit production
	cfg = { # arbitrary
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

	cnf = {
		'rules': defaultdict (lambda: [], {}),
		'lexicon': cfg['lexicon']
	}

	CNF._resolve_unit_production ('VP', 'S', cfg['rules'], cnf)
	CNF._resolve_unit_production ('Z', 'K', cfg['rules'], cnf)
	assert len (cnf['rules']['S']) == 3
	for t in ['Verb PP', 'Verb NP', 'VP PP']: assert t in cnf['rules']['S']

	assert len (cnf['rules']['K']) == 2
	for t in ['Verb PP', 'VP PP']: assert t in cnf['rules']['K']

def test__resolve_unit_production_3 ():
	# when the right-side non-terminal of UP derives one unit production 
	cfg = { # arbitrary
		'rules': {
			'S': ['VP', 'VP PP'],
			'VP': ['K', 'Verb PP', 'Verb NP'],
			'K': ['Z T', 'VP PP', 'Z'], # arbitrary
			'Z': ['N', 'VP PP'], # arbitrary
			'N': ['XX XX', 'MM MM']
		},
		'lexicon': {
			'Verb': ['x', 'y'],
			'PP': ['z', 'y']
		}
	}

	cnf = {
		'rules': defaultdict (lambda: [], {}),
		'lexicon': cfg['lexicon']
	}

	CNF._resolve_unit_production ('VP', 'S', cfg['rules'], cnf)
	for t in ['Verb PP', 'Verb NP', 'XX XX', 'MM MM', 'Z T', 'VP PP']: assert t in cnf['rules']['S']
	for t in ['Verb PP', 'Verb NP', 'XX XX', 'MM MM', 'Z T', 'VP PP']: assert t in cnf['rules']['VP']
	for t in ['XX XX', 'MM MM', 'Z T', 'VP PP']: assert t in cnf['rules']['K']
	for t in ['VP PP', 'XX XX', 'MM MM']: assert t in cnf['rules']['Z']

def test__resolve_unit_production_4 ():
	# when UP derives non-terminal, which has more than one unit production
	cfg = { # arbitrary
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

	cnf = {
		'rules': defaultdict (lambda: [], {}),
		'lexicon': cfg['lexicon']
	}

	CNF._resolve_unit_production ('VP', 'S', cfg['rules'], cnf)
	assert len (cnf['rules']['S']) == 4
	for t in ['Verb PP', 'Adverb Verb PP', 'Z T', 'VP PP']: assert t in cnf['rules']['S']
	for t in ['Verb PP', 'Adverb Verb PP', 'Z T', 'VP PP']: assert t in cnf['rules']['VP']

def test__resolve_more2_nonterminal_ ():
	# there are three or more nonterminals
	cfg = { # arbitrary
		'rules': {
			'S': ['VP', 'VP PP XX'],
			'NP': ['XX YY ZZ TT', 'NM IQ']
		},
	}

	cnf = {
		'rules': defaultdict (lambda: [], {}),
	} 

	d = 0
	d = CNF._resolve_more2_nonterminal (d, 'VP PP XX', 1, 'S', cfg['rules'], cnf)
	d = CNF._resolve_more2_nonterminal (d, 'XX YY ZZ TT', 0, 'NP', cfg['rules'], cnf)

	assert d == 3
	assert cnf['rules']['S'][1] == 'X1 XX'
	assert cnf['rules']['X1'][0] == 'VP PP'
	assert cnf['rules']['NP'][0] == 'X2 TT'
	assert cnf['rules']['X2'][0] == 'X3 ZZ'
	assert cnf['rules']['X3'][0] == 'XX YY'

def test__is_mixed ():
	ri1 = 'XX walk YY'
	ri2 = 'RUNN'
	L = {
		'ADJ': ['lazy'],
		'Verb': ['run', 'walk']
	}

	assert CNF._is_mixed (ri1, L) is True
	assert CNF._is_mixed (ri2, L) is False

# @pytest.mark.skip ()
def test__resolve_mixed ():
	cfg = { # arbitrary
		'rules': {
			'S': ['VP', 'VP PP', 'Verb yy zz'],
			'K': ['zz']
		},
		'lexicon': {
			'Verb': ['x', 'yy'],
			'PP': ['zz', 'y'],

		}
	}

	cnf = {
		'rules': defaultdict (lambda: [], {}),
		'lexicon': cfg['lexicon']
	} 

	dcount = 0
	dcount = CNF._resolve_mixed (dcount, 'Verb yy zz', 2, 'S', cnf['rules'], cfg['rules'], cnf['lexicon'])
	dcount = CNF._resolve_mixed (dcount, 'zz', 0, 'K', cnf['rules'], cfg['rules'], cnf['lexicon'])	

	assert dcount == 2
	cnf['lexicon']['X1'][0] == 'yy'
	cnf['lexicon']['X2'][0] == 'zz'
	cnf['rules']['S'][2] = 'Verb X1 X2'
	cnf['rules']['K'][0] = 'X2'

# @pytest.mark.skip ()
def test_to_CNF (cfg):
	cnf = CNF.to_CNF (cfg)
	assert len (cnf['rules']) == 11
	assert len (cnf['rules']['S']) == 5
	assert len (cnf['lexicon']['S']) == 3
	for nt in cnf['rules']['S']: assert len (nt.split (' ')) == 2
	for nt in cnf['rules']['VP']: assert len (nt.split (' ')) == 2
	for nt in cnf['rules']['NP']: assert len (nt.split (' ')) == 2
	for nt in cnf['rules']['Nominal']: assert len (nt.split (' ')) == 2
	for nt in cnf['rules']['PP']: assert len (nt.split (' ')) == 2
	
