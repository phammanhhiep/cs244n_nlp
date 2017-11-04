import os, sys
sys.path.insert (0, os.getcwd ())
import ngram
Ngram = ngram.Ngram

from collections import defaultdict
import random, math 

class GTd (Ngram):
	def count_fof (): pass
		# count frequency of frequency c

	def smooth_zero_n (self): pass
		# assign value to Nc with zero count

	def adjustedly_count (self, params): pass