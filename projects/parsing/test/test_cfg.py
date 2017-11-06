import os, sys
sys.path.insert (0, os.getcwd ())
import cfg

from collections import defaultdict
import random, math 
import pytest

@pytest.fixture
def CFG ():
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

# @pytest.mark.skip ()
def test__resolve_unit_production_1 ():
	# when the right-side non-terminal of UP derives a terminal
	CFG = { # arbitrary
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

	CNF = {
		'rules': defaultdict (lambda: [], {}),
		'lexicon': CFG['lexicon']
	}

	cfg._resolve_unit_production ('Verb', 'S', CFG['rules'], CNF)
	cfg._resolve_unit_production ('K', 'PP', CFG['rules'], CNF)

	assert len (CNF['lexicon']['S']) == 2
	for i in ['x', 'y']: assert i in CNF['lexicon']['S']
	assert len (CNF['rules']['S']) == 0

	assert len (CNF['lexicon']['PP']) == 4
	for i in ['y', 'z', 'a', 'b']: assert i in CNF['lexicon']['PP']
	assert len (CNF['rules']['PP']) == 2
	for i in ['AB XY', 'AC XY']: assert i in CNF['rules']['PP']

# @pytest.mark.skip ()
def test__resolve_unit_production_1_1 ():
	# when the right-side non-terminal of UP derives a terminal. Special case: all non-terminal in right side derive terminals
	CFG = { # arbitrary
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

	CNF = {
		'rules': defaultdict (lambda: [], {}),
		'lexicon': CFG['lexicon']
	}

	cfg._resolve_unit_production ('K', 'PP', CFG['rules'], CNF)

	assert len (CNF['lexicon']['PP']) == 6
	for i in ['y', 'z', 'a', 'b', 'n', 'm']: assert i in CNF['lexicon']['PP']
	assert len (CNF['rules']['PP']) == 1
	for i in ['AB XY']: assert i in CNF['rules']['PP']

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
		'rules': defaultdict (lambda: [], {}),
		'lexicon': CFG['lexicon']
	}

	cfg._resolve_unit_production ('VP', 'S', CFG['rules'], CNF)
	cfg._resolve_unit_production ('Z', 'K', CFG['rules'], CNF)
	assert len (CNF['rules']['S']) == 3
	for t in ['Verb PP', 'Verb NP', 'VP PP']: assert t in CNF['rules']['S']

	assert len (CNF['rules']['K']) == 2
	for t in ['Verb PP', 'VP PP']: assert t in CNF['rules']['K']

def test__resolve_unit_production_3 ():
	# when the right-side non-terminal of UP derives one unit production 
	CFG = { # arbitrary
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

	CNF = {
		'rules': defaultdict (lambda: [], {}),
		'lexicon': CFG['lexicon']
	}

	cfg._resolve_unit_production ('VP', 'S', CFG['rules'], CNF)
	for t in ['Verb PP', 'Verb NP', 'XX XX', 'MM MM', 'Z T', 'VP PP']: assert t in CNF['rules']['S']
	for t in ['Verb PP', 'Verb NP', 'XX XX', 'MM MM', 'Z T', 'VP PP']: assert t in CNF['rules']['VP']
	for t in ['XX XX', 'MM MM', 'Z T', 'VP PP']: assert t in CNF['rules']['K']
	for t in ['VP PP', 'XX XX', 'MM MM']: assert t in CNF['rules']['Z']

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
		'rules': defaultdict (lambda: [], {}),
		'lexicon': CFG['lexicon']
	}

	cfg._resolve_unit_production ('VP', 'S', CFG['rules'], CNF)
	assert len (CNF['rules']['S']) == 4
	for t in ['Verb PP', 'Adverb Verb PP', 'Z T', 'VP PP']: assert t in CNF['rules']['S']
	for t in ['Verb PP', 'Adverb Verb PP', 'Z T', 'VP PP']: assert t in CNF['rules']['VP']

def test__resolve_more2_nonterminal_ ():
	# there are three or more nonterminals
	CFG = { # arbitrary
		'rules': {
			'S': ['VP', 'VP PP XX'],
			'NP': ['XX YY ZZ TT', 'NM IQ']
		},
	}

	CNF = {
		'rules': defaultdict (lambda: [], {}),
	} 

	d = 0
	d = cfg._resolve_more2_nonterminal (d, 'VP PP XX', 1, 'S', CFG['rules'], CNF)
	d = cfg._resolve_more2_nonterminal (d, 'XX YY ZZ TT', 0, 'NP', CFG['rules'], CNF)

	assert d == 3
	assert CNF['rules']['S'][1] == 'Dummy_1 XX'
	assert CNF['rules']['Dummy_1'][0] == 'VP PP'
	assert CNF['rules']['NP'][0] == 'Dummy_2 TT'
	assert CNF['rules']['Dummy_2'][0] == 'Dummy_3 ZZ'
	assert CNF['rules']['Dummy_3'][0] == 'XX YY'

def test__is_mixed ():
	ri1 = 'XX walk YY'
	ri2 = 'RUNN'
	L = {
		'ADJ': ['lazy'],
		'Verb': ['run', 'walk']
	}

	assert cfg._is_mixed (ri1, L) is True
	assert cfg._is_mixed (ri2, L) is False

# @pytest.mark.skip ()
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

	CNF = {
		'rules': defaultdict (lambda: [], {}),
		'lexicon': CFG['lexicon']
	} 

	dcount = 0
	dcount = cfg._resolve_mixed (dcount, 'Verb yy zz', 2, 'S', CNF['rules'], CFG['rules'], CNF['lexicon'])
	assert dcount == 2
	CNF['lexicon']['Dummy_1'][0] == 'yy'
	CNF['lexicon']['Dummy_2'][0] == 'zz'
	CNF['rules']['S'][2] = 'Verb Dummy_1 Dummy_2'

# @pytest.mark.skip ()
def test_to_CNF (CFG):
	CNF = cfg.to_CNF (CFG)
	assert len (CNF['rules']) == 12
	assert len (CNF['rules']['S']) == 5
	assert len (CNF['lexicon']['S']) == 3
	for nt in CNF['rules']['S']: assert len (nt.split (' ')) == 2
	for nt in CNF['rules']['VP']: assert len (nt.split (' ')) == 2
	for nt in CNF['rules']['NP']: assert len (nt.split (' ')) == 2
	for nt in CNF['rules']['Nominal']: assert len (nt.split (' ')) == 2
	for nt in CNF['rules']['PP']: assert len (nt.split (' ')) == 2
	
