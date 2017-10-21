from collections import defaultdict
import sys, os
sys.path.insert (0, os.getcwd ())
import wordBasedModels.ibmmodel1 as ibmmodel1
import phraseBasedModels.standard as standard
IBMModel1 = ibmmodel1.IBMModel1
Standard = standard.Standard

class LogLinear (Standard): pass

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
	ll = LogLinear (scorpus, tcorpus)
	phrases = ll.extract_phrases (ibm1, max_tphrase_len=max_tphrase_len, max_move=max_move)

	for k,v in phrases.items ():
		if len (v['align'].keys ()) > 0: print (k,v['align'])
