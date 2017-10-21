import os, sys
sys.path.insert (0, os.getcwd ())

from collections import defaultdict
import pytest
import wordBasedModels.ibmmodel1

AER = wordBasedModels.ibmmodel1.AER

def test_aer_get_recall (): pass

def test_aer_get_precision (): pass

def test_aer_get_aer (): pass

def test_aer_get_sure_alignments ():
	expected_alignments = {(3,1): 'S', (2,1): 'P', (1,1): 'S', (4,1): 'S'}
	sure = AER.get_sure_alignments (expected_alignments)
	expected = 3
	assert len (sure) == expected

def test_aer_get_possible_alignments ():
	expected_alignments = {(3,1): 'S', (2,1): 'P', (1,1): 'S', (4,1): 'P'}
	sure = AER.get_possible_alignments (expected_alignments)
	expected = 2
	assert len (sure) == expected	

def test_aer_get_predicted_possible_total ():
	manual_possible_alignments = [(1,1), (2,5), (3,4)]
	alignments = [(1,1), (2,4), (3,4), (4,5)]
	c = AER.get_predicted_possible_total (manual_possible_alignments, alignments)
	expected = 2
	assert c == expected

def test_aer_get_predicted_sure_total ():
	manual_sure_alignments = [(1,1), (2,5), (3,4)]
	alignments = [(1,1), (2,4), (3,4), (4,5)]
	c = AER.get_predicted_sure_total (manual_sure_alignments, alignments)
	expected = 2
	assert c == expected	