from nltk import sent_tokenize, word_tokenize

from collections import defaultdict
import random, heapq

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
				for j in range (l):
					tw = tsent[j]
					parameters['target'][tw]['count'] += self.count_occur (tw, tsent)
					parameters['source'][sw]['align'][tw]['count'] += self.count_alignment (sw, tw, ssent, tsent)

		self.parameters = self.get_p (parameters)

	def get_translation_candidates (self, source_sent, cutoff=3, default_pmi=0.000005):
		translations = {}
		for sw in source_sent:
			translations[sw] = []
			if self.parameters['source'].get (sw, None) is None:
				translations[sw].append ({'NULL_WORD': default_pmi})
			else:
				alignments = self.parameters['source'][sw]['align']
				for tw, v in alignments.items ():
					translations[sw].append ({tw: v['pmi']})
					if len (translations[sw]) > cutoff:
						translations[sw] = heapq.nlargest (cutoff, translations[sw], key=lambda x: list (x.values ())[0])
		return translations

	def align (self, source_sent, target_sent):
		# get the alignment vector
		# assume the training corpora including words in source and target sentence
		alignments = []
		l = len (target_sent)
		m = len (source_sent)

		for i in range (m):
			sw = source_sent[i]
			alignments.append (None)
			for j in range (l):
				tw = target_sent[j]
				alignments[i] = self.get_best_alignment (alignments[i], sw, tw, j)

		return alignments

	def get_best_alignment (self, alignment, sw, tw, cur_index):
		cur_tw = self.parameters['source'][sw]['align'].get (tw, None)
		if cur_tw is None: pass
		else:
			cur_pmi = cur_tw['pmi']
			if alignment is None or (cur_pmi > alignment['pmi']):
				alignment = {'pmi': cur_pmi, 'index': cur_index, 'sw': sw, 'tw': tw}
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

class EM:
	# Estimation maximization
	# using for IBM model 1 and 2
	# Issue: not consider NULL word when collect count and calculate q,t (NEED TO FIX)

	def run (self, source_corpus, target_corpus, get_q=None, get_t=None, get_delta=None, iteration=3, q=None, t=None):
		q = get_q (source_corpus=source_corpus, target_corpus=target_corpus) if q is None else q
		t = get_t () if t is None else t

		for s in range (iteration):
			word_counts = defaultdict (lambda: {'count': 0, 'align': defaultdict (lambda: 0, {})}, {})
			alignment_counts = defaultdict (lambda: {'count': 0, 'align': defaultdict (lambda: 0, {})}, {})
			pair_num = len (source_corpus)
			for k in range (pair_num):
				target_sent = target_corpus[k]
				source_sent = source_corpus[k]
				l = len (target_sent)
				m = len (source_sent)
				for i in range (m):
					sw = source_sent[i]
					for j in range (l):
						tw = target_sent[j]
						delta = get_delta (q, t, target_sent, sw, j, i, m=m, l=l)
						
						# update word count
						word_counts[tw]['count'] += delta
						word_counts[tw]['align'][sw] += delta

						# update index alignment count
						alignment_counts[(i, l, m)]['count'] += delta
						alignment_counts[(i, l, m)]['align'][j] += delta

			# update q, t
			q = get_q (alignment_counts=alignment_counts, q=q)
			t = get_t (word_counts=word_counts, t=t)

		return q,t	

class IBMModel1 ():
	def __init__ (self, source_corpus, target_corpus, EM):
		self.em = EM ()
		self.source_corpus = source_corpus
		self.target_corpus = target_corpus

	@staticmethod
	def get_q (alignment_counts=None, q=None, source_corpus=None, target_corpus=None):
		# calculate the aligment conditional probability
		# FIX. Need to incorporate NULL word
		q = {} if q is None else q
		if alignment_counts is None and source_corpus is not None and target_corpus is not None: # initialization
			pair_num = len (target_corpus)
			for k in range (pair_num):
				target_sent = target_corpus[k]
				source_sent = source_corpus[k]
				l = len (target_sent)
				m = len (source_sent)
				for i in range (m):
					for j in range (l):
						q[(j,i,l,m)] = 1 / (l)
		else: pass
		return q

	@staticmethod
	def get_t (word_counts=None, t=None):
		# calculate the translation conditional probability
		# FIX. Need to incorporate NULL word
		t = {} if (t is None) else t
		if word_counts is None: # initialization
			random.seed ()
			init_value = random.random ()
			t = defaultdict (lambda: defaultdict (lambda: init_value, {}), {})
		else:
			for tw,v in word_counts.items ():
				tw_count = v['count']
				for sw,sw_count in v['align'].items ():
					t[sw][tw] = sw_count / tw_count
		return t 

	@staticmethod	
	def get_delta (q=None, t=None, tsent=None, sw=None, j=None, i=None, m=None, l=None):
		tsent_t_sum = 0
		tsent_len = len (tsent)
		tw = tsent[j]
		for x in range (tsent_len):
			current_tw = tsent[x]
			tsent_t_sum += t[sw][current_tw]
		delta = t[sw][tw] / tsent_t_sum
		return delta

	def get_alignment_p (self, sw, m): pass
		# confitional probability of a single source word p (fj,a|ei)
		# alignment p for a single word only

	def get_translation_p (self): pass

		# translation p for a single word only

	def estimate_translation_parameters (self, iteration=3, t=None, q=None):
		self.q, self.t = self.em.run (self.source_corpus, self.target_corpus, self.get_q, self.get_t, self.get_delta, iteration, q, t)

	def get_translation_candidates (self, source_sent, cutoff=3):
		# assume source_sent is ready for translation, i.e. no need to do any preprocess
		# cutoff: number of candidate picked for each source word
		translations = {}
		for w in source_sent:
			translations[w] = {'candidate': []}
			if self.t.get (w, None) is None: continue
			else:
				for target_word, t_value in self.t[w].items ():
					translations[w]['candidate'].append ({target_word: t_value})
					if len (translations[w]['candidate']) > cutoff:
						translations[w]['candidate'] = heapq.nlargest (cutoff, translations[w]['candidate'], key=lambda x: list (x.values ())[0])
		return translations

	def align (self, source_sent, target_sent):
		# get the alignment vector
		# assume the training corpora including words in source and target sentence
		alignments = []
		l = len (target_sent)
		m = len (source_sent)

		for i in range (m):
			sw = source_sent[i]
			alignments.append (None)
			for j in range (l):
				tw = target_sent[j]
				alignments[i] = self.get_best_alignment (alignments[i], sw, tw, j,i,l,m)

		return alignments

	def get_best_alignment (self, alignment, sw, tw, j,i,l,m, null_p=0.00001):
		# p = p (a,fi|ej) = t 
		p = self.t[sw].get (tw, -1)
		if p == -1: pass
		else:
			if alignment is None or (p > alignment['p']):
				alignment = {'p': p, 'sw': sw, 'tw': tw, 'index': j}
		if alignment is None and (j == l-1):
			alignment = {'p': null_p, 'sw': sw, 'tw': '_NULL_', 'index': -1}
		return alignment

	def get_alignment_indices (self, alignments):
		i = []
		for a in alignments:
			i.append (a['index'])
		return i

	def get_alignment_words (self, alignments):
		w = []
		for a in alignments:
			w.append (a['tw'])
		return w

	def evaluate (self, alignments, expected_alignments):
		def _get_recall (): pass
		def _get_precision (): pass
		def _get_total_sure (): pass
		def _get_total_possible (): pass
		
		alen = len (alignments)
		sure = _get_total_sure (expected_alignments)
		possible = _get_total_possible (expected_alignments)
		r, s_matched_total = _get_recall (alignments, expected_alignments)
		pr, pr_matched_total = _get_precision (alignments, expected_alignments)
		aer = 1 - (s_matched_total + pr_matched_total) / (sure + alen)

		return r,pr,aer

class IBMModel2 (IBMModel1):
	def __init__ (self, source_corpus, target_corpus, EM):
		IBMModel1.__init__ (self, source_corpus, target_corpus, EM)
	
	@staticmethod
	def get_q (alignment_counts=None, q=None, source_corpus=None, target_corpus=None):
		if alignment_counts is None: # initialization
			q = IBMModel1.get_q (source_corpus=source_corpus, target_corpus=target_corpus)
		else:	
			for k,v in alignment_counts.items ():
				for j,count in v['align'].items ():
					kj = list (k)
					kj.insert (0, j)
					q[tuple (kj)] = count / v['count']
		return q

	@staticmethod	
	def get_delta (q=None, t=None, tsent=None, sw=None, j=None, i=None, m=None, l=None):
		qt_sum = 0
		tsent_len = len (tsent)
		tw = tsent[j]
		for x in range (tsent_len):
			current_tw = tsent[x]
			qt_sum += t[sw][current_tw] * q[(x, i, l, m)]
		delta = t[sw][tw] * q[(j, i, l, m)]/ qt_sum
		return delta

	def get_best_alignment (self, alignment, sw, tw, j,i,l,m, null_p=0.00001):
		# p = p (a,fi|ej) = t * q
		# j: a target word index, i: a source word index, l: target sentence length, m: source sentence length
		# alignment: current best alignment of the source word
		# sw: source word, tw: target word
		# null_p: p to align a source word with NULL word

		t = self.t[sw].get (tw, -1)
		q = self.q.get((j,i,l,m), -1)
		if t == -1 or q == -1: pass
		else:
			p = t * q
			if alignment is None or (p > alignment['p']):
				alignment = {'p': p, 'sw': sw, 'tw': tw, 'index': j}
		if alignment is None and (j == l-1):
			alignment = {'p': null_p, 'sw': sw, 'tw': "_NULL_", 'index': -1}			
		return alignment 

class IBMModel3 (IBMModel2): pass

class IBMModel4 (): pass

class IBMModel5 (): pass

if __name__ == '__main__':
	import tools.sample_data as sd
	# sample_data_dir = 'data/bg-en-sample/'
	# sample_data_dir = 'data/fr-en-sample/'
	sample_data_dir = 'data/vn-en-sample/'

	# data_dir = 'source_data/bg-en/'
	# data_dir = 'source_data/fr-en/'

	# target = data_dir + 'europarl-v7.bg-en.en'
	# source = data_dir + 'europarl-v7.bg-en.bg'
	# target = data_dir + 'europarl-v7.fr-en.en'
	# source = data_dir + 'europarl-v7.fr-en.fr'

	target_sample = sample_data_dir + 'en'
	# source_sample = sample_data_dir + 'bg'
	# source_sample = sample_data_dir + 'fr'
	source_sample = sample_data_dir + 'vn'

	# sd.sample_data (source, target, source_sample, target_sample, 1500)

	input_ss = "Tôi có thể sống với những tiêu chuẩn tối thiểu này, nhưng tôi sẽ yêu cầu Ủy ban giám sát tình hình một cách cẩn thận."
	input_ts = "I can live with these minimum standards, but I would ask the Commission to monitor the situation very carefully."

	scorpus = []
	tcorpus = []

	with open (target_sample, encoding='utf8') as efd, open (source_sample, encoding='utf8') as ffd:
		tcorpus = [word_tokenize (i) for i in efd]
		scorpus = [word_tokenize (i) for i in ffd]

	input_ss = word_tokenize(input_ss)
	input_ts = word_tokenize(input_ts)

	ibm1 = None
	ibm2 = None

	# #PMI
	# pmi = PMI (scorpus, tcorpus)
	# pmi.estimate_translation_parameters ()
	# alignments = pmi.align (input_ss, input_ts)
	# avector = pmi.get_alignment_indices (alignments)
	# awords = pmi.get_alignment_words (alignments)
	# print (avector)
	# print (awords)
	# print (input_ss)
	# print (input_ts)
	# print ('----------- END -----------')

	# IBM Model 1
	ibm1 = IBMModel1 (scorpus, tcorpus, EM)
	ibm1.estimate_translation_parameters (iteration=5)
	alignments = ibm1.align (input_ss, input_ts)
	avector = ibm1.get_alignment_indices (alignments)
	awords = ibm1.get_alignment_words (alignments)
	# print (ibm1.t)
	# print (alignments)
	print (avector)
	print (awords)
	print (input_ss)
	print (input_ts)
	print ('----------- END -----------')

	# IBM model 2
	if ibm1 is not None: default_t = ibm1.t
	else: default_t = None
	ibm2 = IBMModel2 (scorpus, tcorpus, EM)
	ibm2.estimate_translation_parameters (iteration=5, t=default_t)
	alignments = ibm2.align (input_ss, input_ts)
	avector = ibm2.get_alignment_indices (alignments)
	awords = ibm2.get_alignment_words (alignments)
	# print (ibm2.t)	
	# print (alignments)
	print (avector)
	print (awords)
	print (input_ss)
	print (input_ts)
	print ('----------- END -----------')

