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
			'NP': ['Pronoun', 'Proper-Noun', 'Det Nominal'],
			'Nominal': ['Noun', 'Nominal Noun', 'Nominal PP'],
			'VP': ['Verb', 'Verb NP', 'Verb NP PP', 'Verb PP', 'Verb NP NP','VP PP'],
			'PP': ['Preposition NP']
		},
		'lexicon': {
			'Det': ['that','the', 'a', ],
			'Noun': ['book', 'flight', 'meal', 'money', 'dinner'],
			'Verb': ['book', 'include', 'prefer', ],
			'Pronoun': ['I','she','me', 'you'],
			'Proper-Noun': ['Houston', 'NWA'],
			'Aux': ['does','can'],
			'Preposition': ['from','to','on','near','through', ]
		}
	}

@pytest.fixture
def p_table ():
	return {
		'rules': {
			'S': [0.8,0.15,0.05],
			'NP': [0.35, 0.3, 0.2],
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

def test__collect_pos (up_CNF):
	words = ['book', 'the', 'flight', 'through', 'Houston']
	wnum = len (words)
	G, p_table = up_CNF
	table = PCKY._gen_parse_table (words)
	for j in range (1, wnum+1):
		PCKY._collect_pos (j, words, G, table, p_table)

	assert len (table[0][1]) == 2
	nt_list = [list (l.keys())[0] for l in table[0][1]]
	p_list = [0.3, 0.1]
	for i in ['Verb', 'Noun']: 
		assert i in nt_list
		ii = nt_list.index (i)
		table[0][1][ii]['p'] = p_list[ii]


# @pytest.mark.skip ()
def test_recognize (up_CNF):
	words = ['book', 'the', 'flight', 'through', 'Houston']
	G, p_table = up_CNF
	table = PCKY.recognize (words, G, p_table)

	assert len (table[0][5]) == 10
	nt_list = [list (l.keys())[0] for l in table[0][5]]
	for i in ['S', 'VP', 'X2', 'TX']: assert i in nt_list

def test_parse (): pass

def test_evaluate (): pass


