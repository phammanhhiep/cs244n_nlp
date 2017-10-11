from collections import defaultdict
import random, heapq

import sys, os
sys.path.insert (0, os.getcwd ())
import word_alignment.ibmmodel1
# IBMModel1 = word_alignment.ibmmodel1.IBMModel1
IBMModel1 = word_alignment.ibmmodel1.IBMModel1S

class IBMModel2 (IBMModel1):
	def __init__ (self, source_corpus, target_corpus):
		IBMModel1.__init__ (self, source_corpus, target_corpus)
	
	@staticmethod
	def get_q (alignment_counts=None, q=None, source_corpus=None, target_corpus=None, ignore_null=False):
		if alignment_counts is None: # initialization
			q = IBMModel1.get_q (source_corpus=source_corpus, target_corpus=target_corpus, ignore_null=ignore_null)
		else:	
			for k,v in alignment_counts.items ():
				for j,count in v['align'].items ():
					kj = list (k)
					kj.insert (0, j)
					q[tuple (kj)] = count / v['count']
		return q

	@staticmethod	
	def get_delta (q=None, t=None, tsent=None, sw=None, j=None, i=None, m=None, l=None, ignore_null=False):
		if j == 0 and (not ignore_null): # for the NULL word
			# delta = 1 / (len (tsent) + 1)]
			delta = 1 / 200000 
		else: 
			qt_sum = 0
			tsent_len = len (tsent)
			tw = tsent[j]
			for x in range (tsent_len):
				current_tw = tsent[x]
				qt_sum += t[sw][current_tw] * q[(x, i, l, m)]
			delta = t[sw][tw] * q[(j, i, l, m)]/ qt_sum
		return delta

	def get_best_alignment (self, alignment, sw, tw, j,i,l,m, null_p=0.00001, ignore_null=False):
		# p = p (a,fi|ej) = t * q
		# j: a target word index, i: a source word index, l: target sentence length, m: source sentence length
		# alignment: current best alignment of the source word
		# sw: source word, tw: target word
		# null_p: p to align a source word with NULL word
		# use index of -1 to indicate sw is aligned with NULL word and sw is not in the training corpora.
		# if ignore_null is False, then index of -1 and 0 are both refer to the NULL word

		t = self.t[sw].get (tw, -1)
		q = self.q.get((j,i,l,m), -1)
		if t == -1 or q == -1: pass
		if j == 0 and ignore_null: return alignment
		else:
			p = t * q
			if alignment is None or (p > alignment['p']):
				alignment = {'p': p, 'sw': sw, 'tw': tw, 'index': j}
		if alignment is None and (j == l-1):
			alignment = {'p': null_p, 'sw': sw, 'tw': "_NULL_", 'index': -1}			
		return alignment 

if __name__ == '__main__':
	import tools.sample_data as sd
	from nltk import sent_tokenize, word_tokenize
	
	# data_dir = 'source_data/bg-en/'
	data_dir = 'source_data/fr-en/'

	# target = data_dir + 'europarl-v7.bg-en.en'
	# source = data_dir + 'europarl-v7.bg-en.bg'
	target = data_dir + 'europarl-v7.fr-en.en'
	source = data_dir + 'europarl-v7.fr-en.fr'

	sample_data_dir = 'data/fr-en-sample/'
	# sample_data_dir = 'data/bg-en-sample/'
	# sample_data_dir = 'data/vn-en-sample/'	

	target_sample = sample_data_dir + 'en'
	source_sample = sample_data_dir + 'fr'
	# source_sample = sample_data_dir + 'bg'
	# source_sample = sample_data_dir + 'vn'

	sd.sample_data (source, target, source_sample, target_sample, 500)

	# input_ss = "Tôi có thể sống với những tiêu chuẩn tối thiểu này, nhưng tôi sẽ yêu cầu Ủy ban giám sát tình hình một cách cẩn thận."
	# input_ts = "I can live with these minimum standards, but I would ask the Commission to monitor the situation very carefully."

	# input_ss = "Monsieur le Président, nous débattons une fois de plus de la politique européenne de concurrence."
	# input_ts = "Mr President, once again we are debating the European Union' s competition policy."

	input_ss = "Le sixième rapport de la Commission européenne tire des conclusions très précieuses."
	input_ts = "The European Commission' s sixth report presents very valuable conclusions."

	scorpus = []
	tcorpus = []

	with open (target_sample, encoding='utf8') as efd, open (source_sample, encoding='utf8') as ffd:
		tcorpus = [word_tokenize (i) for i in efd]
		scorpus = [word_tokenize (i) for i in ffd]

	input_ss = word_tokenize(input_ss)
	input_ts = word_tokenize(input_ts)
	ignore_null = False
	null_num = 2
	ibm1 = None

	# IBM Model 1
	ibm1 = IBMModel1 (scorpus, tcorpus)
	ibm1.estimate_translation_parameters (iteration=5, ignore_null=ignore_null, null_num=null_num)
	default_t = None if ibm1 is None else ibm1.t

	# IBM model 2
	ibm2 = IBMModel2 (scorpus, tcorpus)
	ibm2.estimate_translation_parameters (iteration=5, t=default_t, ignore_null=ignore_null, null_num=null_num)
	alignments = ibm2.align (input_ss, input_ts, ignore_null=ignore_null)
	avector = ibm2.get_alignment_indices (alignments)
	awords = ibm2.get_alignment_words (alignments)
	# print (ibm2.t)	
	# print (alignments)
	print (avector)
	print (awords)
	print (input_ss)
	print (input_ts)
	print ('----------- END -----------')