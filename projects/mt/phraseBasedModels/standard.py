from collections import defaultdict
import sys, os
sys.path.insert (0, os.getcwd ())
import wordBasedModels.ibmmodel1 as ibmmodel1
IBMModel1 = ibmmodel1.IBMModel1

class Standard:
	def __init__ (self, source_corpus=None, target_corpus=None):
		self.source_corpus = source_corpus
		self.target_corpus = target_corpus		

	def extract_phrases (self, word_aliner, max_tphrase_len=3, max_move=3):
		# iterate all possible English phrases to find their aligned phrases
		phrases = defaultdict (lambda: {'count': 0, 'align': defaultdict (lambda: {'count': 0},{})}, {})
		pair_num = len (self.source_corpus)

		for k in range (pair_num):
			ssent = self.source_corpus[k]
			tsent = self.target_corpus[k]
			wa = word_aliner.align (ssent, tsent)
			wa = word_aliner.get_alignment_indices (wa)
			tsent_len = len (tsent)
			ssent_len = len (ssent)
			phrases = self._extract_phrases_by_sent (phrases, ssent, tsent, wa, max_tphrase_len, max_move)
			phrases = self.estimate_phrase_probability (phrases)
		return phrases

	def _extract_phrases_by_sent (self, phrases, ssent, tsent, wa, max_tphrase_len=3, max_move=3):
		tsent_len = len (tsent)
		ssent_len = len (ssent)
		for i in range (tsent_len):
			ei_start = i + 1 # not consider NULL word
			max_end_index = (i + max_tphrase_len) if (i + max_tphrase_len) < tsent_len else tsent_len
			for j in range (i, max_end_index):
				ei_end = j + 1 # not consider NULL word
				fj_start, fj_end = ssent_len, 0 # initialization
				for ei, fj in wa:
					if ei >= ei_start and ei <= ei_end:
						fj_start = min (fj_start, fj)
						fj_end = max (fj_end, fj)
				phrase_e, phrase_f = self._extract_phrase (fj_start, fj_end, ei_start, ei_end, ssent, tsent, wa, max_move)
				phrases = self._count_phrases (phrases, phrase_e, phrase_f)
		return phrases

	def _extract_phrase (self, fj_start, fj_end, ei_start, ei_end, ssent, tsent, wa, max_move=0):
		# return phrase e and a corresponding list of phrases f as a tuple
		phrase_e = tsent[ei_start-1:ei_end]
		phrase_e = ' '.join (phrase_e)
		phrase_f = []
		if fj_end == 0: pass
		else:
			# check if any invalid source index in the middle of the source index range
			for ei,fj in wa:
				if (fj >= fj_start and fj <= fj_end) and (ei < ei_start or ei > ei_end):
					return phrase_e, phrase_f

			fj_s = fj_start
			backward_count = 0
			while True and backward_count <= max_move:
				forward_count = 0
				fj_e = fj_end
				while True and forward_count <= max_move:
					phrase_fj = ssent[fj_s-1:fj_e]
					phrase_fj = ' '.join (phrase_fj)
					phrase_f.append (phrase_fj)
					fj_e += 1
					forward_count += 1
					if fj_e > len (ssent) or self._is_aligned_index (fj_e, wa): break
				fj_s += -1
				backward_count += 1
				if fj_s == 0 or self._is_aligned_index (fj_s, wa): break
		return phrase_e, phrase_f

	def _is_aligned_index (self, fj, wa):
		for ei,fk in wa:
			if fk == fj: return True
		return False				

	def _count_phrases (self, phrases, phrase_e, phrase_f):
		# count number of times phrase_e aligned
		# count number of times (sentences) phrase_fi aligned with phrase_e
		phrases[phrase_e]['count'] += 1
		pf_len = len (phrase_f)
		if pf_len == 0: pass
		else:
			fractional_count = 1 / pf_len
			for pfi in phrase_f:
				phrases[phrase_e]['align'][pfi]['count'] += fractional_count
		return phrases	

	def estimate_phrase_probability (self, phrases):
		for ephrase, ev in phrases.items ():
			e_count = ev['count']
			for fphrase, fv in ev['align'].items ():
				fv['tranp'] = fv['count'] / e_count
		return phrases

	def estimate_reordering_probability (self): pass

	def _set_orientation (self): pass
		# determine if a phrase is of one of three orientation types

	def _smooth_orientation (self): pass

	def decoding (self, source_sent, phrase_tran_table, stack_size=None, stack_alpha=None, best_n=None):
		translation_options = self.get_translation_options (source_sent, phrase_tran_table)
		future_cost_table = self.get_future_cost_table (translation_options)
		stacks = []
		stack_num = len (source_sent)
		for s in range (stack_num):
			word_num = s + 1
			stacks[s] = []
			prev_stack = self._get_prev_stack (word_num, stacks) 
			new_hypotheses = self.create_hypotheses (word_num, translation_options)
			expanded_hypotheses = self.expanded_hypotheses (prev_stack)
			stacks[s].extend (new_hypotheses)
			stacks[s].extend (expanded_hypotheses)
			stacks[s] = self.get_translation_cost (stacks[s], future_cost_table)
			stacks[s] = self.prune_hypo (stacks[s], stack_size, stack_alpha)
		best_translations = self.get_n_best_translation (stacks, best_n)
		return best_translations
				
	def _get_prev_stack (self, word_num, stacks): pass	

	def get_translation_options (self, source_sent, phrase_tran_table): pass		

	def create_hypotheses (self, word_num, translation_options): pass

	def expand_hypotheses (self): pass

	def get_future_cost (self): pass

	def prune_hypo (self, stack, stack_size, stack_alpha):
		stack = self.recombine_hypo (stack)
		stack = self.threshold_prune (stack, stack_alpha)
		stack = self.histogram_prune (stack, stack_size)
		return stack

	def recombine_hypo (self, stack): pass

	def threshold_prune (self, stack, stack_alpha): pass

	def histogram_prune (self, stack, stack_size): pass

	def get_future_cost_table (self, translation_options): pass

	def get_translation_cost (self, stack, future_cost_table): pass
			
	def get_n_best_translation (stacks, best_n): pass


if __name__ == '__main__':
	from nltk import sent_tokenize, word_tokenize
	from bs4 import BeautifulSoup

	data_dir =  'source_data/HLTNAACL_2003/fr-en/English-French.training/English-French/training/'
	target_file = data_dir + 'hansard.36.1.house.debates.00{}.e'
	source_file = data_dir + 'hansard.36.1.house.debates.00{}.f'
	scorpus = []
	tcorpus = []

	filenumber = 1

	for i in range (filenumber):
		with open (target_file.format (i+1)) as efd, open (source_file.format (i+1)) as ffd:
			tcorpus.extend (word_tokenize (j) for j in efd)
			scorpus.extend (word_tokenize (j) for j in ffd)

	# def extract_trial_sentences (b):
	# 	d = {}
	# 	sents = b.find_all ('s')
	# 	num = len (sents)
	# 	for i in range (num):
	# 		s =sents[i]
	# 		d[s['snum']] = word_tokenize(s.text.strip (' '))
	# 	return d

	# def get_trial_alignments (alignments, sent_num):
	# 	new_alignments = defaultdict (lambda: {}, {})
	# 	num = len (alignments)
	# 	for i in range (num):
	# 		ai = alignments[i]
	# 		confidence = ai[3]
	# 		line = ai[0]
	# 		new_alignments[line][(int (ai[1]), int (ai[2]))] = confidence
	# 	return new_alignments

	# trail_source_file = 'source_data/HLTNAACL_2003/fr-en/English-French.trial/English-French/trial/trial.f'
	# trail_target_file = 'source_data/HLTNAACL_2003/fr-en/English-French.trial/English-French/trial/trial.e'
	# trail_alignment_file = 'source_data/HLTNAACL_2003/fr-en/English-French.trial/English-French/trial/trial.wa'

	# trial_scorpus = []
	# trial_tcorpus = []
	# trial_alignment = []

	# with open (trail_target_file) as efd, open (trail_source_file) as ffd, open (trail_alignment_file) as afd:
	# 	trial_target = BeautifulSoup (efd, "lxml")
	# 	trail_source = BeautifulSoup (ffd, "lxml")
	# 	trial_tcorpus = extract_trial_sentences (trial_target)
	# 	trial_scorpus = extract_trial_sentences (trail_source)
	# 	trial_alignment = [word_tokenize (i) for i in afd]
	# 	trial_alignment = get_trial_alignments (trial_alignment, len (trial_scorpus))

	# trial_scorpus = [v for k,v in trial_scorpus.items ()]
	# trial_tcorpus = [v for k,v in trial_tcorpus.items ()]	

	print ('- Phase-based Standard -')
	max_tphrase_len = 3
	max_move = 3
	ibm1 = IBMModel1 (scorpus, tcorpus)
	ibm1.estimate_translation_parameters (iteration=5)
	standard = Standard (scorpus, tcorpus)
	phrases = standard.extract_phrases (ibm1, max_tphrase_len=max_tphrase_len, max_move=max_move)

	# for k,v in phrases.items ():
	# 	if len (v['align'].keys ()) > 0: print (k,v['align'])