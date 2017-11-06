import os, sys
sys.path.insert (0, os.getcwd ())
import cky, cfg

from collections import defaultdict
import random, math 
import pytest

@pytest.fixture
def CFG ():
	return {
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

@pytest.fixture
def CNF (CFG):
	CNF = cfg.to_CNF (CFG)
	return CNF	

def test__create_parse_table (): pass

def test__collect_pos (CNF):
	words = ['book', 'the', 'flight', 'through', 'Houston']
	wnum = len (words)
	G = CNF
	table = [[None for i in range (wnum + 1)] for j in range (wnum + 1)]
	for j in range (1, wnum+1):
		cky._collect_pos (j, words, G, table)

	assert len (table[0][1]) == 5
	for i in ['S', 'VP', 'Nominal', 'Verb', 'Noun']: assert i in table[0][1]
	assert len (table[2][3]) == 2
	for i in ['Noun', 'Nominal']: assert i in table[2][3]
	assert len (table[3][4]) == 1
	for i in ['Preposition']: assert i in table[3][4]


# @pytest.mark.skip ()
def test__collect_constituents (CNF):
	words = ['book', 'the', 'flight', 'through', 'Houston']
	wnum = len (words)
	G = CNF
	table = [[None for i in range (wnum + 1)] for j in range (wnum + 1)]
	for j in range (1, wnum+1):
		cky._collect_pos (j, words, G, table)
		cky._collect_constituents (j, words, G, table)

	assert True

def test_recognize (CNF, CFG):
	words = ['book', 'the', 'flight', 'through', 'Houston']
	G = CNF
	# G = CFG
	table = cky.recognize (words, G)
	print (table)
	assert True
def test_parse (): pass

