import os, sys
sys.path.insert (0, os.getcwd ())
import ngram
Ngram = ngram.Ngram

from collections import defaultdict
import random, math 

class Backoff (Ngram):
	def __init__ (self, corpus=None):
		Ngram.__init__ (self, corpus)

	def count (self, sent, params, ngram=2):	
		# collect count of all N-grams, given a N value
		# sent: a list of word
		start_index = 0
		end_index = 0
		word_num = len (sent)
		for i in range (word_num):
			start_index = i
			end_index = start_index + ngram - 1
			if end_index >= word_num: break
			w = sent [end_index]
			params[w]['count'] += 1
			params[w]['single'] = True
			for j in range (end_index - start_index):
				history = ' '.join (sent [start_index+j:end_index])
				if (ngram-j) > 2 or history == '<s>': params[history]['count'] += 1 
				params[w][ngram-j][history]['count'] += 1
		return params

	def estimate_logp (self, params, ngram=2):
		for w, w_v in params.items ():
			if w_v['single'] is True:
				for history, his_v in w_v[ngram].items ():
					join_count = his_v['count']
					lower_gram = ngram - 1
					while join_count == 0 and lower_gram > 0:
						if lower_gram > 1: 
							lower_history = history.split (' ')[lower_gram-1]
							join_count = w_v[lower_gram][lower_history]['count']
						else:
							lower_history = history
							join_count = w_v[lower_history]['count']  
						lower_gram -= 1
					his_count = params[history]['count']
					p = join_count / his_count
					his_v['logp'] = math.log (p)
		return params			

if __name__ == '__main__':
	from nltk import word_tokenize
	from bs4 import BeautifulSoup
	data_dir =  '../mt/source_data/HLTNAACL_2003/fr-en/English-French.training/English-French/training/'
	target_file = data_dir + 'hansard.36.1.house.debates.00{}.e'
	tcorpus = []

	filenumber = 1

	for i in range (filenumber):
		with open (target_file.format (i+1)) as efd:
			tcorpus.extend (word_tokenize (j) for j in efd)

	def extract_trial_sentences (b):
		d = []
		sents = b.find_all ('s')
		num = len (sents)
		for i in range (num):
			s =sents[i]
			d.append (word_tokenize(s.text.strip (' ')))
		return d

	trail_target_file = '../mt/source_data/HLTNAACL_2003/fr-en/English-French.trial/English-French/trial/trial.e'

	trial_tcorpus = []

	with open (trail_target_file) as efd:
		trial_target = BeautifulSoup (efd, "lxml")
		trial_tcorpus = extract_trial_sentences (trial_target)

	print ('--- Ngram ---')
	ngram = 3	
	m = Ngram (tcorpus)
	params = m.train (ngram)
	pp = m.evaluate (trial_tcorpus, params, ngram)
	print (pp)
	# c=0
	# for k,v in params.items ():
	# 	print (k,v)
	# 	c += 1
	# 	if c == 3: break

	print ('--- Back off ---')
	ngram = 3	
	m = Backoff (tcorpus)
	params = m.train (ngram)
	pp = m.evaluate (trial_tcorpus, params, ngram)
	print (pp)
	# c=0
	# for k,v in params.items ():
	# 	print (k,v)
	# 	c += 1
	# 	if c == 3: break