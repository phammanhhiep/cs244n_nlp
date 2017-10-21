from nltk import sent_tokenize, word_tokenize
from collections import defaultdict
import random

class PMI ():
	# Poinwise mutual information
	# Assume one target word can align with any number of source words. One source word must align with exactly one target word.
	# Assume corpora and input sentences are processed and ready for alignment analysis

	def __init__ (self, source_corpus, target_corpus):
		self.source_corpus = source_corpus
		self.target_corpus = target_corpus

	def count_occur (self, word, sent):
		count = 0
		for w in sent:
			if (w == word): count += 1
		return count		

	def count_alignment (self, sword, tword, ssent, tsent):
		if (sword in ssent) and (tword in tsent): return 1
		else: return 0

	def get_priori_p (self, count, total):
		return count / total
		
	def get_join_p (self, count, total):
		return count / total
		
	def get_pmi (self, spriori_p, tpriori_p, join_p):
		pmi = join_p / (spriori_p * tpriori_p)
		return pmi

	def get_p (self, parameters):
		sw_count = parameters['_stat']['source']['word_count']
		tw_count = parameters['_stat']['target']['word_count']
		alignment_count = parameters['_stat']['source']['alignment_count']

		for sw,vi in parameters['source'].items ():
			vi['priori_p'] = self.get_priori_p (vi['count'], sw_count)
			for tw,vj in parameters['target'].items ():
				vj['priori_p'] = self.get_priori_p (vj['count'], tw_count)
				vi['align'][tw]['join_p']  = self.get_join_p (vi['align'][tw]['count'], alignment_count)
				vi['align'][tw]['pmi'] = self.get_pmi (vi['priori_p'], vj['priori_p'], vi['align'][tw]['join_p'])
		return parameters

	def estimate_translation_parameters (self):
		sent_num = len (self.source_corpus)

		parameters = {
			'source': defaultdict (lambda: {'align': defaultdict (lambda: {'count': 0, 'join_p': 0, 'pmi': 0}, {}), 'count': 0, 'priori_p': 0}, {}),
			'target': defaultdict (lambda: {'count': 0, 'priori_p': 0}, {}),
			'_stat': {
				'source': {'word_count': 0, 'alignment_count': 0},
				'target': {'word_count': 0},
			}			
		}

		for k in range (sent_num):
			ssent = self.source_corpus[k]
			tsent = self.target_corpus[k]
			l = len (tsent)
			m = len (ssent)
			parameters['_stat']['source']['word_count'] += m
			parameters['_stat']['target']['word_count'] += l
			parameters['_stat']['source']['alignment_count'] += (l * m)

			for i in range (m):
				sw = ssent[i]
				parameters['source'][sw]['count'] += self.count_occur (sw, ssent)
				for j in range (l + 1):
					if j == 0:
						tw = '_NULL_'
						parameters['target'][tw]['count'] += 1
						parameters['source'][sw]['align'][tw]['count'] += 1						
					else:
						tw = tsent[j-1]
						parameters['target'][tw]['count'] += self.count_occur (tw, tsent)
						parameters['source'][sw]['align'][tw]['count'] += self.count_alignment (sw, tw, ssent, tsent)

		self.parameters = self.get_p (parameters)

	def align (self, source_sent, target_sent):
		# get the alignment vector
		# assume the training corpora including words in source and target sentence
		alignments = []
		target_sent.insert (0, '_NULL_')
		l = len (target_sent)
		m = len (source_sent)

		for i in range (m):
			sw = source_sent[i]
			alignments.append (None)
			for j in range (l):
				tw = target_sent[j]
				alignments[i] = self.get_best_alignment (alignments[i], sw, tw, j,i,l,m)

		return alignments

	def get_best_alignment (self, alignment, sw, tw, j,i,l,m, null_pmi=1):
		cur_tw = self.parameters['source'][sw]['align'].get (tw, None)
		if cur_tw is None: pass
		else:
			cur_pmi = cur_tw['pmi']
			if alignment is None or (cur_pmi > alignment['pmi']):
				alignment = {'pmi': cur_pmi, 'index': j, 'sw': sw, 'tw': tw}
		if alignment is None and (j == l-1):
			alignment = {'pmi': null_pmi, 'index': 0, 'sw': sw, 'tw': '_NULL_'}
		return alignment		

	def get_alignment_indices (self, alignments):
		i = []
		for a in alignments:
			i.append (a['index'])
		return i

	def get_alignment_words (self, alginments):
		w = []
		for a in alignments:
			w.append (a['tw'])
		return w

if __name__ == '__main__':
	import sys, os
	sys.path.insert (0, os.getcwd ())
	import tools.sample_data as sd
	
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

	sd.sample_data (source, target, source_sample, target_sample, 200)

	# input_ss = "Tôi có thể sống với những tiêu chuẩn tối thiểu này, nhưng tôi sẽ yêu cầu Ủy ban giám sát tình hình một cách cẩn thận."
	# input_ts = "I can live with these minimum standards, but I would ask the Commission to monitor the situation very carefully."

	input_ss = "C'est ce que nous demandons aujourd'hui au commissaire."
	input_ts = "This is what we are today asking the Commissioner for."


	scorpus = []
	tcorpus = []

	with open (target_sample, encoding='utf8') as efd, open (source_sample, encoding='utf8') as ffd:
		tcorpus = [word_tokenize (i) for i in efd]
		scorpus = [word_tokenize (i) for i in ffd]

	input_ss = word_tokenize(input_ss)
	input_ts = word_tokenize(input_ts)

	#PMI
	pmi = PMI (scorpus, tcorpus)
	pmi.estimate_translation_parameters ()
	alignments = pmi.align (input_ss, input_ts)
	avector = pmi.get_alignment_indices (alignments)
	awords = pmi.get_alignment_words (alignments)
	# print (pmi.parameters[])
	print (avector)
	print (awords)
	print (input_ss)
	print (input_ts)
	print ('----------- END -----------')