from collections import defaultdict
import random, heapq

import sys, os
sys.path.insert (0, os.getcwd ())
import word_alignment.ibmmodel1
IBMModel1 = word_alignment.ibmmodel1.IBMModel1
# IBMModel1 = word_alignment.ibmmodel1.Smoothed
# IBMModel1 = word_alignment.ibmmodel1.Heuristic


class IBMModel2 (IBMModel1):
	def __init__ (self, source_corpus, target_corpus):
		IBMModel1.__init__ (self, source_corpus, target_corpus)

	def get_q (self, alignment_counts=None, q=None):
		if alignment_counts is None: # initialization
			q = IBMModel1.get_q (self)
		else:	
			for k,v in alignment_counts.items ():
				for j,count in v['align'].items ():
					kj = list (k)
					kj.insert (0, j)
					q[tuple (kj)] = count / v['count']
		return q

	def get_delta (self, q=None, t=None, tsent=None, sw=None, j=None, i=None, m=None, l=None):
		qt_sum = 0
		tsent_len = len (tsent)
		tw = tsent[j]
		for x in range (tsent_len):
			current_tw = tsent[x]
			qt_sum += t[sw][current_tw] * q[(x, i + 1, l, m)]
		delta = t[sw][tw] * q[(j, i + 1, l, m)]/ qt_sum
		return delta


	# Deprecated. LIKELY to be incorrect	
	# def get_best_alignment (self, alignment, sw, tw, j,i,l,m, null_p=0.00001):
	# 	# p = p (a,fi|ej) = t * q
	# 	# j: a target word index, i: a source word index, l: target sentence length, m: source sentence length
	# 	# alignment: current best alignment of the source word
	# 	# sw: source word, tw: target word
	# 	# null_p: p to align a source word with NULL word
	# 	# Souce sentence has no NULL, and so no position 0. That is why (i + 1) is used

	# 	q = {x: self.q.get((x,i + 1,l,m), -1) for x in range (l)} 
	# 	t = self.t[sw].get (tw, -1)
	# 	if t == -1: pass
	# 	else:
	# 		p = [{'value': v * t, 'j': k} for k,v in q.items () if v > -1]
	# 		for pi in p:
	# 			if alignment is None or (pi['value'] > alignment['p']):
	# 				alignment = {'p': pi['value'], 'sw': sw, 'tw': tw, 'index': pi['j']}
	# 	if alignment is None and (j == l-1):
	# 		alignment = {'p': null_p, 'sw': sw, 'tw': "_NULL_", 'index': 0}			
	# 	return alignment

	def get_best_alignment (self, alignment, sw, tw, j,i,l,m, null_p=0.00001):
		# p = p (a,fi|ej) = t * q
		# j: a target word index, i: a source word index, l: target sentence length, m: source sentence length
		# alignment: current best alignment of the source word
		# sw: source word, tw: target word
		# null_p: p to align a source word with NULL word
		# Souce sentence has no NULL, and so no position 0. That is why (i + 1) is used

		q = self.q.get((j,i + 1,l,m), -1) 
		t = self.t[sw].get (tw, -1)
		if t == -1 or q == -1: pass
		else:
			p = q * t
			if alignment is None or (p > alignment['p']):
				alignment = {'p': p, 'sw': sw, 'tw': tw, 'index': j}
		if alignment is None and (j == l-1):
			alignment = {'p': null_p, 'sw': sw, 'tw': "_NULL_", 'index': 0}			
		return alignment	

if __name__ == '__main__':
	import sys, os
	sys.path.insert (0, os.getcwd ())
	from nltk import sent_tokenize, word_tokenize
	from bs4 import BeautifulSoup

	data_dir =  'source_data/HLTNAACL_2003/fr-en/English-French.training/English-French/training/'
	target_file = data_dir + 'hansard.36.1.house.debates.0{}.e'
	source_file = data_dir + 'hansard.36.1.house.debates.0{}.f'
	scorpus = []
	tcorpus = []

	filenumber = 1

	for i in range (filenumber):
		file_index = ('{}' if (i + 1) >= 10 else '0{}').format (i+1) 
		with open (target_file.format (file_index)) as efd, open (source_file.format (file_index)) as ffd:
			tcorpus.extend (word_tokenize (j) for j in efd)
			scorpus.extend (word_tokenize (j) for j in ffd)

	def extract_trial_sentences (b):
		d = {}
		sents = b.find_all ('s')
		num = len (sents)
		for i in range (num):
			s =sents[i]
			d[s['snum']] = word_tokenize(s.text.strip (' '))
		return d

	def get_trial_alignments (alignments, sent_num):
		new_alignments = defaultdict (lambda: {}, {})
		num = len (alignments)
		for i in range (num):
			ai = alignments[i]
			confidence = ai[3]
			line = ai[0]
			new_alignments[line][(int (ai[1]), int (ai[2]))] = confidence
		return new_alignments

	trail_source_file = 'source_data/HLTNAACL_2003/fr-en/English-French.trial/English-French/trial/trial.f'
	trail_target_file = 'source_data/HLTNAACL_2003/fr-en/English-French.trial/English-French/trial/trial.e'
	trail_alignment_file = 'source_data/HLTNAACL_2003/fr-en/English-French.trial/English-French/trial/trial.wa'

	trial_scorpus = []
	trial_tcorpus = []
	trial_alignment = []

	with open (trail_target_file) as efd, open (trail_source_file) as ffd, open (trail_alignment_file) as afd:
		trial_target = BeautifulSoup (efd, "lxml")
		trail_source = BeautifulSoup (ffd, "lxml")
		trial_tcorpus = extract_trial_sentences (trial_target)
		trial_scorpus = extract_trial_sentences (trail_source)
		trial_alignment = [word_tokenize (i) for i in afd]
		trial_alignment = get_trial_alignments (trial_alignment, len (trial_scorpus))

	# print ('- IBM Model 1 Heuristic -')
	print ('- IBM Model 1 -')
	null_num = 1
	init_null = None
	# exponent = 3
	ibm1 = IBMModel1 (scorpus, tcorpus)
	# ibm1.get_init_t (exponent=exponent)
	ibm1.estimate_translation_parameters (iteration=5, null_num=null_num, init_null=init_null)
	aer, recall, precision = ibm1.evaluate (trial_scorpus, trial_tcorpus, trial_alignment)
	print (aer, recall, precision)
	print ('----END ----')

	print ('- IBM Model 2 -')
	null_num = 5
	init_null = None
	exponent = 3
	ibm2 = IBMModel2 (scorpus, tcorpus)
	# ibm2.get_init_t (exponent=exponent)
	ibm2.estimate_translation_parameters (iteration=5, t=ibm1.t, null_num=null_num, init_null=init_null)
	aer, recall, precision = ibm2.evaluate (trial_scorpus, trial_tcorpus, trial_alignment)
	print (aer, recall, precision)
	print ('----END ----')