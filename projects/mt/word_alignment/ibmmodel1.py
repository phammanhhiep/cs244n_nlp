from collections import defaultdict
import random, math

class AER:
	@staticmethod
	def evaluate (predicted_sure_num, predicted_possible_num, alignment_total, sure_total):
		r = AER.get_recall (predicted_sure_num, sure_total)
		p = AER.get_precision (predicted_possible_num, alignment_total)
		aer = AER.get_aer (predicted_possible_num, predicted_sure_num, alignment_total, sure_total)
		return aer, r, p

	@staticmethod
	def get_recall (predicted_sure_num, sure_total):
		return predicted_sure_num / sure_total

	@staticmethod
	def get_precision (predicted_possible_num, alignment_total):
		return predicted_possible_num / alignment_total

	@staticmethod	
	def get_aer (predicted_possible_num, predicted_sure_num, alignment_total, sure_total):
		return 1 - (predicted_sure_num + predicted_possible_num) / (alignment_total + sure_total)

	@staticmethod
	def get_possible_alignments (expected_alignments):
		# Evaluate one sentence
		# Get possible alignment from the manual alignment
		possible = []
		for k,v in expected_alignments.items ():
			if v == 'P': possible.append (k)
		return possible

	@staticmethod
	def get_sure_alignments (expected_alignments):
		# Evaluate one sentence
		# Get sure alignment from the manual alignment
		sure = []
		for k,v in expected_alignments.items ():
			if v == 'S': sure.append (k)
		return sure

	@staticmethod	
	def get_predicted_sure_total (manual_sure_alignments, alignments):
		# Evaluate one sentence pair
		found = [i for i in alignments if i in manual_sure_alignments]
		return len (found)

	@staticmethod	
	def get_predicted_possible_total (manual_possible_alignments, alignments):
		# Evaluate one sentence pair
		found = [i for i in alignments if i in manual_possible_alignments]
		return len (found)		

class IBMModel1 ():
	def __init__ (self, source_corpus, target_corpus, evaluation=AER):
		self.evaluation = evaluation
		self.source_corpus = source_corpus
		self.target_corpus = target_corpus
	
	def get_q (self, alignment_counts=None, q=None, source_corpus=None, target_corpus=None, default_q=0.1):
		# calculate the aligment conditional probability
		# default_q is p for unseen (j,i,l,m) tuple

		q = defaultdict (lambda: default_q, {}) if q is None else q
		if alignment_counts is None and source_corpus is not None and target_corpus is not None: # initialization
			pair_num = len (target_corpus)
			for k in range (pair_num):
				target_sent = target_corpus[k]
				source_sent = source_corpus[k]
				target_sent = ['_NULL_'] + target_sent # add NULL word
				l = len (target_sent)
				m = len (source_sent)
				for i in range (m):
					for j in range (l):
						q[(j,i+1,l,m)] = 1 / (l) # Source sentence has no null word, their index is i + 1
		else: pass
		return q

	def get_t (self, word_counts=None, t=None, init_t=None, null_num=1, init_null=None):
		# calculate the translation conditional probability
		t = {} if (t is None) else t
		if word_counts is None: # initialization
			random.seed ()
			init_value = random.random ()
			t = defaultdict (lambda: defaultdict (lambda: init_value, {}), {})
		else:
			for tw,v in word_counts.items ():
				tw_count = v['count']
				for sw, align_count in v['align'].items ():
					if tw == '_NULL_':
						t[sw][tw] = null_num * align_count / tw_count
					else:
						t[sw][tw] = align_count / tw_count
		return t 

	def get_delta (self, q=None, t=None, tsent=None, sw=None, j=None, i=None, m=None, l=None):
		tsent_t_sum = 0
		tsent_len = len (tsent)
		tw = tsent[j]
		for x in range (tsent_len):
			current_tw = tsent[x]
			tsent_t_sum += t[sw][current_tw]
		delta = t[sw][tw] / tsent_t_sum
		return delta

	def em_count (self, target_corpus, source_corpus, q, t, get_delta):
		word_counts = defaultdict (lambda: {'count': 0, 'align': defaultdict (lambda: 0, {})}, {})
		alignment_counts = defaultdict (lambda: {'count': 0, 'align': defaultdict (lambda: 0, {})}, {})
		pair_num = len (source_corpus)
		for k in range (pair_num):
			target_sent = target_corpus[k]
			source_sent = source_corpus[k]
			null_words = ['_NULL_']
			target_sent = null_words + target_sent 
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
					alignment_counts[(i + 1, l, m)]['count'] += delta
					alignment_counts[(i + 1, l, m)]['align'][j] += delta

		return word_counts, alignment_counts

	def estimate_translation_parameters (self, iteration=3, t=None, q=None, null_num=1, init_t=None, init_null=None):
		self.q = self.get_q (source_corpus=self.source_corpus, target_corpus=self.target_corpus) if q is None else q
		self.t = self.get_t (init_t=init_t) if t is None else t

		for s in range (iteration):
			word_counts, alignment_counts = self.em_count (self.target_corpus, self.source_corpus, self.q, self.t, self.get_delta)
			self.q = self.get_q (alignment_counts=alignment_counts, q=self.q)
			self.t = self.get_t (word_counts=word_counts, t=self.t, null_num=null_num, init_null=init_null)

	def align (self, source_sent, target_sent):
		# get the alignment vector
		# assume the training corpora including words in source and target sentence
		alignments = []
		target_sent = ['_NULL_'] + target_sent # for NULL word
		l = len (target_sent)
		m = len (source_sent)

		for i in range (m):
			sw = source_sent[i]
			alignments.append (None)
			for j in range (l):
				tw = target_sent[j]
				alignments[i] = self.get_best_alignment (alignments[i], sw, tw, j,i,l,m)

		return alignments

	# NeedFix: optimize null_p	
	def get_best_alignment (self, alignment, sw, tw, j,i,l,m, null_p=0.00001):
		# p = p (a,fi|ej) = t 
		# use index of -1 to indicate sw is aligned with NULL word and sw is not in the training corpora.
		# if ignore_null is False, then index of -1 and 0 are both refer to the NULL word

		p = self.t[sw].get (tw, -1)
		if p == -1: pass
		else:
			if alignment is None or (p > alignment['p']):
				alignment = {'p': p, 'sw': sw, 'tw': tw, 'index': j}
		if alignment is None and (j == l-1):
			alignment = {'p': null_p, 'sw': sw, 'tw': '_NULL_', 'index': 0}
		return alignment

	def get_alignment_indices (self, alignment):
		# get alignment indices of a sentence pair
		# Source sentence has no NULL word. So word indices are i + 1
		indices = []
		anum = len (alignment)
		for i in range (anum):
			a = alignment[i]
			indices.append ((a['index'], i+1)) 
		return indices

	def get_alignment_words (self, alignment):
		# get alignment words of a sentence pair
		w = []
		for a in alignment:
			w.append (a['tw'])
		return w

	def evaluate (self, sources, targerts, manual_alignments):
		pair_num = len (sources)
		predicted_sure_total = predicted_possible_total = manual_sure_total = alignment_total = 0
		for sent_index ,source_sent in sources.items ():
			target_sent = targerts[sent_index]
			manual_alignment = manual_alignments[sent_index]

			alignment = self.align (source_sent, target_sent)
			alignment_indices = self.get_alignment_indices (alignment)
			alignment_total += len (alignment_indices)
			manual_possible = self.evaluation.get_possible_alignments (manual_alignment)
			manual_sure = self.evaluation.get_sure_alignments (manual_alignment)	
			manual_sure_total += len (manual_sure)	
			predicted_possible_total += self.evaluation.get_predicted_possible_total (manual_possible, alignment_indices)
			predicted_sure_total += self.evaluation.get_predicted_sure_total (manual_sure, alignment_indices)	

		aer, recall, precision = self.evaluation.evaluate (predicted_sure_total, predicted_possible_total, alignment_total, manual_sure_total)
		return aer, recall, precision

class Smoothed (IBMModel1):
	# smoothing with uniform distribution to deal with rare words
	# initialize using LLS
	def __init__ (self, source_corpus, target_corpus, added_count=0.01, v_wc=12, v_bonus=100):
		# v_wc: assumed average length of a source sentence
		# v_bonus: to make sure v > source vocabulary
		# added_count: addictive count number 	
		IBMModel1.__init__ (self, source_corpus, target_corpus)
		self.added_count = added_count
		self.V = len (self.source_corpus) * v_wc + v_bonus
	

	def get_t (self, word_counts=None, t=None, init_t=None, null_num=1, init_null=None):
		# calculate the translation conditional probability
		# NeedToFix: Need to assign probability mass to unseen word
		t = {} if (t is None) else t
		if word_counts is None: # initialization
			random.seed ()
			init_value = random.random ()		
			t = defaultdict (lambda: defaultdict (lambda: init_value, {}), {})
		else:
			for tw,v in word_counts.items ():
				tw_count = v['count']
				for sw, align_count in v['align'].items ():
					if tw == '_NULL_':
						t[sw][tw] = null_num * align_count / tw_count
					else:					
						t[sw][tw] = (align_count + self.added_count) / (tw_count + self.added_count * self.V) # Smoothing
		return t	

class Heuristic (IBMModel1):
	def _get_wc (self):	
		# get word count of each source word and the coocurrence with each of its target words	
		sent_num = len (self.source_corpus)

		parameters = {
			'target': defaultdict (lambda: {'align': defaultdict (lambda: {'count': 0,'llr': None, 'llr_p': None}, {}), 'count': 0, 'llr_sum': 0}, {}),
			'source': defaultdict (lambda: {'count': 0, 'priori_p': 0}, {}),
			'_stat': {
				'source': {'word_count': 0},
				'target': {'largest_llr_sum': 0}
			}			
		}

		for k in range (sent_num):
			ssent = self.source_corpus[k]
			tsent = self.target_corpus[k]
			null_words = ['_NULL_']
			tsent = null_words + tsent 			
			l = len (tsent)
			m = len (ssent)
			parameters['_stat']['source']['word_count'] += m

			for i in range (m):
				sw = ssent[i]
				parameters['source'][sw]['count'] += 1
				for j in range (l):					
					tw = tsent[j]
					parameters['target'][tw]['count'] += 1
					parameters['target'][tw]['align'][sw]['count'] += 1

		return parameters

	# NeedFix: remove negative llr
	# NeedFix: Add n NULL word	
	def _get_llr (self, parameters, exponent=None):
		# get log likelihood ratio of each word pair
		# assign the unigram p of a source word for LLR of NULL and the source word
		largest_llr_sum = 0
		for tw, tw_v in parameters['target'].items ():
			tw_count = tw_v['count']
			for sw, align_v in tw_v['align'].items ():
				sw_priori_p = parameters['source'][sw]['count'] / parameters['_stat']['source']['word_count']
				if tw == '_NULL_':
					align_v['llr'] = align_v['count'] * math.log (sw_priori_p) 
				else:
					sw_cond_tw_p = align_v['count'] / tw_count # p (fi|ei)
					align_v['llr'] = align_v['count'] * math.log (sw_cond_tw_p / sw_priori_p)
				align_v['llr'] = self._llr_to_exp (align_v['llr'], exponent)
				tw_v['llr_sum'] += align_v['llr']
			largest_llr_sum = largest_llr_sum if largest_llr_sum > tw_v['llr_sum'] else tw_v['llr_sum']
		parameters['_stat']['target']['largest_llr_sum'] = largest_llr_sum
		return parameters	

	def _llr_to_exp (self, llr, exponent=3):
		# raise llr to an exponent
		return math.pow (llr, exponent)

	def _llr_to_p (self, parameters):
		# convert log likelihood ratio to corresponding probability
		largest_llr_sum = parameters['_stat']['target']['largest_llr_sum']
		for tw, tw_v in parameters['target'].items ():
			for sw, align_v in tw_v['align'].items ():
				align_v['llr_p'] = align_v['llr'] / largest_llr_sum
		return parameters

	def get_init_t (self, exponent=3):
		parameters = self._get_wc ()
		parameters = self._get_llr (parameters, exponent)
		parameters = self._llr_to_p (parameters)
		return parameters

	# NeedFix: review when to init_null occurs in the code. Likely when init t	
	def get_t (self, word_counts=None, t=None, init_t=None, init_null=None, null_num=1):
		# calculate the translation conditional probability
		# NeedToFix: Need to assign probability mass to unseen word
		t = {} if (t is None) else t
		if word_counts is None: # initialization
			random.seed ()
			init_value = random.random ()		
			t = defaultdict (lambda: defaultdict (lambda: init_value, {}), {})
			if init_t is not None:
				for tw, tw_v in init_t['target'].items():
					for sw, align_v in tw_v['align'].items():
						t[sw][tw] = align_v['llr_p']
		else:
			for tw,v in word_counts.items ():
				tw_count = v['count']
				for sw, align_count in v['align'].items ():
					if tw == '_NULL_':
						if init_null is not None: pass
						else:
							t[sw][tw] = null_num * align_count / tw_count
					else:
						t[sw][tw] = align_count / tw_count
		return t

class Combined (IBMModel1): pass

if __name__ == '__main__':
	import sys, os
	sys.path.insert (0, os.getcwd ())
	from nltk import sent_tokenize, word_tokenize
	from bs4 import BeautifulSoup

	data_dir =  'source_data/HLTNAACL_2003/fr-en/English-French.training/English-French/training/'
	target_file = data_dir + 'hansard.36.1.house.debates.00{}.e'
	source_file = data_dir + 'hansard.36.1.house.debates.00{}.f'
	scorpus = []
	tcorpus = []

	filenumber = 5

	for i in range (filenumber):
		with open (target_file.format (i+1)) as efd, open (source_file.format (i+1)) as ffd:
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

	print ('- Model 1 Standard -')
	ibm1 = IBMModel1 (scorpus, tcorpus)
	ibm1.estimate_translation_parameters (iteration=2)
	aer, recall, precision = ibm1.evaluate (trial_scorpus, trial_tcorpus, trial_alignment)
	print (aer, recall, precision)
	print ('----END ----')

	print ('- Model 1 Smoothed -')
	added_count = 0.01
	v_wc=10
	v_bonus=100
	null_num = 3
	smoothed = Smoothed (scorpus, tcorpus, added_count, v_wc, v_bonus)
	smoothed.estimate_translation_parameters (iteration=4, null_num=null_num)
	aer, recall, precision = smoothed.evaluate (trial_scorpus, trial_tcorpus, trial_alignment)
	print (aer, recall, precision)
	print ('----END ----')

	print ('- Model 1 Heuristic -')
	null_num = 3
	init_null = None
	exponent = 3
	heu = Heuristic (scorpus, tcorpus)
	init_t = heu.get_init_t (exponent=exponent)
	heu.estimate_translation_parameters (iteration=4, null_num=null_num, init_t=init_t, init_null=init_null)
	aer, recall, precision = heu.evaluate (trial_scorpus, trial_tcorpus, trial_alignment)
	print (aer, recall, precision)
	print ('----END ----')