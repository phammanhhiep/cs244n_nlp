import os, sys
sys.path.insert (0, os.getcwd ())
import pcfg.pcky, pcfg.pcnf

from collections import defaultdict
import random, math 
import pytest

PCKY = pcfg.pcky.PCKY

@pytest.fixture
def cfg ():
	return {
		'rules': {
			'S': ['NP VP', 'Aux NP VP', 'VP'],
			'NP': ['Pronoun', 'Proper-Noun', 'Det Nominal', 'Nominal'],
			'Nominal': ['Noun', 'Nominal Noun', 'Nominal PP'],
			'VP': ['Verb', 'Verb NP', 'Verb NP PP', 'Verb PP', 'Verb NP NP','VP PP'],
			'PP': ['Preposition NP']
		},
		'lexicon': {
			'Det': ['that','the', 'a'],
			'Noun': ['book', 'flight', 'meal', 'money', 'dinner'],
			'Verb': ['book', 'include', 'prefer'],
			'Pronoun': ['I','she','me', 'you'],
			'Proper-Noun': ['Houston', 'NWA'],
			'Aux': ['does','can'],
			'Preposition': ['from','to','on','near','through']
		}
	}

@pytest.fixture
def p_table ():
	return {
		'rules': {
			'S': [0.8,0.15,0.05],
			'NP': [0.35, 0.3, 0.2,0.15],
			'Nominal': [0.75,0.2,0.05],
			'VP': [0.35, 0.2, 0.1,0.15,0.05,0.15],
			'PP': [1]
		},
		'lexicon': {
			'Det': [0.1, 0.6, 0.3],
			'Noun': [0.1, 0.7, 0.015, 0.085, 0.1],
			'Verb': [0.3, 0.3, 0.4],
			'Pronoun': [0.4, 0.05, 0.15, 0.4],
			'Proper-Noun': [0.6, 0.4],
			'Aux': [0.6, 0.4],
			'Preposition': [0.3, 0.3, 0.2, 0.15, 0.05]
		}
	}	

@pytest.fixture
def up_CNF (cfg, p_table):
	cnf, p_table = pcfg.pcnf.PCNF.to_CNF (cfg, p_table)
	return cnf, p_table

# @pytest.mark.skip ()
def test__collect_pos (up_CNF):
	words = ['book', 'the', 'flight', 'through', 'Houston']
	wnum = len (words)
	G, p_table = up_CNF
	table = PCKY._gen_parse_table (words)
	for j in range (1, wnum+1):
		PCKY._collect_pos (j, words, G, table, p_table)

	assert len (table[0][1]) == 2
	nt_list = [l['head'] for l in table[0][1]]
	p_list = [0.3, 0.1]
	for i in ['Verb', 'Noun']: 
		assert i in nt_list
		ii = nt_list.index (i)
		table[0][1][ii]['p'] = p_list[ii]


def test__estimate_p1 (): pass
	# estimate correct p for pos

def test__estimate_p2 (): pass
	# estimate correct p for unit production

def test__estimate_p3 (): pass
	# estimate correct p for normal rule

# @pytest.mark.skip ()
def test__collect_up ():
	G = { # arbitrary
		'rules': {
			'S': ['K Verb', 'PP Verb'],
			'PP': ['K', 'AB XY'],
			'TT': ['K', 'XX Y'],
			'K': ['Z', 'AC XY'],
			'T': ['Verb', 'TZ OM']
		},
		'lexicon': {
			'Verb': ['x', 'y'],
			'PP': ['z', 'y'],
			'Z': ['a', 'b']
		}
	}

	p_table = { # arbitrary
		'rules': {
			'S': [0.4, 0.6],
			'PP': [0.5, 0.5],
			'TT': [0.2, 0.8],
			'K': [0.1, 0.9],
			'T': [0.9, 0.1]
		},
		'lexicon': {
			'Verb': [0.2, 0.8],
			'PP': [0.6, 0.4],
			'Z': [0.3, 0.7]
		}
	}

	pointer = {'head': 'Z', 'left': None, 'right': None, 'p': 0.3} 
	p1 = PCKY._collect_up ('Z', G['rules'], p_table, pointer)
	pointer = {'head': 'Verb', 'left': None, 'right': None, 'p': 0.8} 
	p2 = PCKY._collect_up ('Verb', G['rules'], p_table, pointer)

	assert len (p1) == 3
	assert len (p2) == 1

# @pytest.mark.skip ()
def test__collect_rules ():
	G = { # arbitrary
		'rules': {
			'S': ['K Verb', 'PP Verb'],
			'PP': ['K', 'AB XY'],
			'TT': ['K', 'XX Y'],
			'K': ['Z', 'AC XY'],
			'T': ['Verb', 'TZ OM']
		},
		'lexicon': {
			'Verb': ['x', 'y'],
			'PP': ['z', 'y'],
			'Z': ['a', 'b']
		}
	}

	p_table = { # arbitrary
		'rules': {
			'S': [0.4, 0.6],
			'PP': [0.5, 0.5],
			'TT': [0.2, 0.8],
			'K': [0.1, 0.9],
			'T': [0.9, 0.1]
		},
		'lexicon': {
			'Verb': [0.2, 0.8],
			'PP': [0.6, 0.4],
			'Z': [0.3, 0.7]
		}
	}

	left_pointer = {'head': 'Z', 'left': None, 'right': None, 'p': 0.3}
	right_pointer = {'head': 'Verb', 'left': None, 'right': None, 'p': 0.8}
	pointers = PCKY._collect_rules (left_pointer, right_pointer, G['rules'], p_table)

	assert len (pointers) == 2
	assert pointers[0]['left']['head'] == 'K'
	assert pointers[0]['right']['head'] == 'Verb'
	assert pointers[0]['head'] == 'S'

	assert pointers[1]['left']['head'] == 'PP'
	assert pointers[1]['right']['head'] == 'Verb'
	assert pointers[1]['head'] == 'S'

# @pytest.mark.skip ()
def test_recognize (up_CNF):
	words = ['book', 'the', 'dinner', 'flight']
	G, p_table = up_CNF
	table = PCKY.recognize (words, G, p_table)
	assert True

	# TEST later

	# assert len (table[0][4]) == 10
	# nt_list = [list (l.keys())[0] for l in table[0][5]]
	# for i in ['S', 'VP', 'X2', 'TX']: assert i in nt_list

def test_parse (): pass

def test_evaluate (): pass


