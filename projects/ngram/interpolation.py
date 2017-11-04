import os, sys
sys.path.insert (0, os.getcwd ())
import ngram, backoff
Ngram = ngram.Ngram
Backoff = backoff.Backoff

from collections import defaultdict
import random, math 

class Interpolation (Backoff):
	def __init__ (self, corpus):
		Backoff.__init__ (self, corpus)

	def em (self): pass
		# train interpolation lambda

	def estimate_logp (self, params, ngram=2, lambda_params=None):
		for w, w_v in params.items ():
			if w_v['single'] is True:
				for history, his_v in w_v[ngram].items ():
					inter_p = 0
					for i in range (ngram):
						lower_gram = ngram - i
						if lower_gram > 1: 
							lower_history = ' '.join (history.split (' ')[-(lower_gram-1):])
							join_count = w_v[lower_gram][lower_history]['count']
							lower_his_count = params[lower_history]['count']	
							lambda_i = lambda_params[-(i+1)]
							p = (lambda_i * join_count) / lower_his_count							
						else:
							p = params[w]['count'] / params['<TOTAL>']['count']
						inter_p += p
					his_v['logp'] = math.log (inter_p)
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
	params = m.estimate_logp (params, ngram)
	pp = m.evaluate (trial_tcorpus, params, ngram)
	print (pp)

	print ('--- Back off ---')
	ngram = 3	
	m = Backoff (tcorpus)
	params = m.train (ngram)
	params = m.estimate_logp (params, ngram)
	pp = m.evaluate (trial_tcorpus, params, ngram)
	print (pp)

	print ('--- Interpolation ---')
	ngram = 3
	# lambda_params = [1/ngram] * ngram
	lambda_params = [0.1, 0.3, 0.6]
	m = Interpolation (tcorpus)
	params = m.train (ngram)
	params = m.estimate_logp (params, ngram, lambda_params)
	pp = m.evaluate (trial_tcorpus, params, ngram)
	print (pp)