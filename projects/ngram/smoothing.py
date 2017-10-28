import os, sys
sys.path.insert (0, os.getcwd ())
import ngram
Ngram = ngram.Ngram

from collections import defaultdict
import random, math 

class AddOne (Ngram):
	def __init__ (self, corpus):
		Ngram.__init__ (self, corpus)

	def estimate_logp (self, params, ngram=2):
		# avoid estimate logp before smoothing
		return params

	def smooth (self, params, ngram=2):
		V = len ([k for k,v in params.items () if v['single'] is True]) # vocabulary size
		for w, w_v in params.items ():
			if w_v['single'] is True:
				for history, his_v in w_v[ngram].items ():
					join_count = his_v['count']
					his_count = params[history]['count']
					p = (join_count + 1) / (his_count + V)
					his_v['logp'] = math.log (p)
		return params

class AddK (AddOne):
	def __init__ (self, corpus, K=1):
		AddOne.__init__ (self, corpus)
		self.K = K

	def smooth (self, params, ngram=2):
		V = len ([k for k,v in params.items () if v['single'] is True]) # vocabulary size
		for w, w_v in params.items ():
			if w_v['single'] is True:
				for history, his_v in w_v[ngram].items ():
					join_count = his_v['count']
					his_count = params[history]['count']
					p = (join_count + self.K) / (his_count + V * self.K)
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

	print ('--- AddOne ---')
	ngram = 3	
	m = AddOne (tcorpus)
	params = m.train (ngram)
	params = m.smooth (params, ngram)
	pp = m.evaluate (trial_tcorpus, params, ngram)
	print (pp)

	print ('--- Add K ---')
	ngram = 3
	K = 0.000001	
	m = AddK (tcorpus, K)
	params = m.train (ngram)
	params = m.smooth (params, ngram)
	pp = m.evaluate (trial_tcorpus, params, ngram)
	print (pp)