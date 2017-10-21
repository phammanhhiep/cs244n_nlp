from collections import defaultdict
import random, heapq

import sys, os
sys.path.insert (0, os.getcwd ())
import word_alignment.ibmmodel2
IBMModel2 = word_alignment.ibmmodel2.IBMModel2

class IBMModel3 (IBMModel2):
	def __init__ (self, source_corpus, target_corpus):
		self.source_corpus = source_corpus
		self.target_corpus = target_corpus

	def sample_alignments (self, e, f): pass
		# do pedgging, hill climb, and get neighbor alignments
	
	def hill_climb (self, pegged_j): pass
		# return best alignment among list of an alignments, including a predefined alignment and its neighbors

	def get_neighbor_alignments (self, a, pegged_j): pass
		# move and swap
