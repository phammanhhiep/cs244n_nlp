import os, sys
sys.path.insert (0, os.getcwd ())
import cfg.cky, cfg.cnf

from collections import defaultdict
import random, math 
import pytest

CKY = cfg.cky.CKY

@pytest.fixture
def CFG ():
	return {
		'rules': {
			'S': ['NP VP', 'Aux NP VP', 'VP'],
			'NP': ['Pronoun', 'Proper-Noun', 'Det Nominal'],
			'Nominal': ['Noun', 'Nominal Noun', 'Nominal PP'],
			'VP': ['Verb', 'Verb NP', 'Verb NP PP', 'Verb PP', 'VP PP'],
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
	CNF = cfg.cnf.CNF.to_CNF (CFG)
	return CNF

@pytest.fixture
def up_CNF (CFG):
	# CNF without handling up
	CNF = cfg.cnf.CNF.to_CNF (CFG, up=False)
	return CNF

def test__create_parse_table (): pass

def test__collect_pos (CNF):
	words = ['book', 'the', 'flight', 'through', 'Houston']
	wnum = len (words)
	G = CNF
	table = CKY._gen_parse_table (words)
	for j in range (1, wnum+1):
		CKY._collect_pos (j, words, G, table)
	assert len (table[0][1]) == 5
	nt_list = [list (l.keys())[0] for l in table[0][1]]
	for i in ['S', 'VP', 'Nominal', 'Verb', 'Noun']: assert i in nt_list
	assert len (table[2][3]) == 2
	nt_list = [list (l.keys())[0] for l in table[2][3]]
	for i in ['Noun', 'Nominal']: assert i in nt_list
	assert len (table[3][4]) == 1
	nt_list = [list (l.keys())[0] for l in table[3][4]]
	for i in ['Preposition']: assert i in nt_list

@pytest.mark.skip ()
def test_recognize_1 (CNF):
	# CNF
	words = ['book', 'the', 'flight', 'through', 'Houston']
	G = CNF
	table = CKY.recognize (words, G)

	assert len (table[0][5]) == 7
	nt_list = [list (l.keys())[0] for l in table[0][5]]
	for i in ['S', 'VP', 'X2']: assert i in nt_list


# @pytest.mark.skip ()
def test_recognize_2 (up_CNF):
	# CNF without handling unit production
	words = ['book', 'the', 'flight', 'through', 'Houston']
	up_CNF['rules']['TX'] = ['VP', 'XX YY']
	G = up_CNF
	table = CKY.recognize (words, G)

	assert len (table[0][5]) == 10
	nt_list = [list (l.keys())[0] for l in table[0][5]]
	for i in ['S', 'VP', 'X2', 'TX']: assert i in nt_list 

def test__handle_up (up_CNF):
	words = ['book', 'the', 'flight', 'through', 'Houston']
	up_CNF['rules']['TX'] = ['VP', 'XX YY']
	G = up_CNF
	table = [[None, [{'VP': [1,2,3,4,5,6]}]], []]
	CKY._handle_up (table, 0, 1, 'VP', G['rules'])
	nt_list = [list (l.keys ())[0]for l in table[0][1]]

	for i in ['S', 'TX', 'VP']: assert i in nt_list


def test_parse (): pass


def test_evaluate (): pass
