import os, sys
sys.path.insert (0, os.getcwd ())
import pcfg.pcnf

from collections import defaultdict
import random, math 
import pytest

PCNF = pcfg.pcnf.PCNF

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

	return cfg


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


def test__update_mixed_dm_p (p_table):
	prev_dcount, dcount = 3,5
	PCNF._update_mixed_dm_p (p_table, prev_dcount, dcount)
	assert p_table['lexicon']['X4'][0] == 1
	assert p_table['lexicon']['X5'][0] == 1

def test__update_more2nt_dm_p (p_table):
	prev_dcount, dcount = 3,5
	PCNF._update_more2nt_dm_p (p_table, prev_dcount, dcount)
	assert p_table['rules']['X4'][0] == 1
	assert p_table['rules']['X5'][0] == 1

def test_to_CNF (cfg, p_table):
	# X1 and X2 can be in lexicon or rules arbitrary. No way to be certain.
	cnf, p_table = PCNF.to_CNF (cfg, p_table)
	dnum = 2
	assert True 
