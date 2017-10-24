from collections import defaultdict
import random, heapq

import sys, os
sys.path.insert (0, os.getcwd ())
import word_alignment.ibmmodel2
IBMModel2 = word_alignment.ibmmodel2.IBMModel2

class IBMModel3:
	def __init__ (self, source_corpus, target_corpus):
		self.source_corpus = source_corpus
		self.target_corpus = target_corpus

	def sample_alignments (self, e, f, aligner):
		# do pedgging, hill climb, and get neighbor alignments
		# expect NULL word in e sentence
		l = len (e)
		m = len (f)
		A = [] # alignments
		best_a_old = aligner.align (f, e).get_alignment_indices ()
		for j in range (l):
			for i in range (m):
				best_a = []; best_a.extend (best_a_old)
				best_a[j] = (best_a[j][0], i + 1)
			best_a = self.hill_climb (best_a, j, aligner)
			A.append (self.neighboring (best_a, j, aligner))
		return A

	def hill_climb (self, a, pegged_j, aligner):
		# return best alignment among list of an alignments, including a predefined alignment and its neighbors
		def _same_alignment (a1, a2):
			# assume ai = (i,j), indices of word ei, and fj
			num = len (a1)
			for n in range (num):
				i1,j1 = a1[n]
				i2,j2 = a2[n]
				if j1 == j2 and i1 != i2:
					return False
			return True

		while True:
			old_a = a
			neighbors = self.neighboring (a, pegged_j)
			for na in neighbors:
				neighbor_p = aligner.get_alignment_p (na)
				a_p = aligner.get_alignment_p (a)
				if neighbor_p > a_p: a = na
			if _same_alignment (a, old_a): break
		return a

	def neighboring (self, a, pegged_j, l, m):
		# move and swap
		# l: len of e; m: len of f
		N = []
		for j in range (l): # move one
			for i in range (m):
				if j != pegged_j:
					a_prime = []; a_prime.extend (a)
					a_prime[j] = (a_prime[j][0], i + 1)
					N.append (a_prime)

		for j1 in range (l): # swap one
			for j2 in range (l):
				if j1 != pegged_j and j2 != pegged_j and j2 != j1:
					a_prime = []; a_prime.extend (a)
					temp = a_prime[j1][1]
					a_prime[j1] = (a_prime[j1][0], a_prime[j2][1])
					a_prime[j2] = (a_prime[j2][0], temp)
					N.append (a_prime)
		return N

	def train (self): pass
		# return the parameters of the model

