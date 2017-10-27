from collections import defaultdict
import random, math 

class Ngram:
	# Estimate probability of N-gram with MLE
	# Support unigram, bigram, and trigram
	def __init__ (self, corpus=None):
		self.corpus = corpus

	def preprocess (self, sent, ngram=2):
		# add <s> and </s> at the begining and end of a sentence respectively
		for i in range (ngram - 1):
			sent.insert (0, '<s>')
		sent.append ('</s>')
		return sent

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
			history = ' '.join (sent [start_index:end_index])
			params[w]['count'] += 1
			params[w]['single'] = True
			if ngram > 2 or history == '<s>': params[history]['count'] += 1 
			params[w][ngram][history]['count'] += 1
		return params

	def train (self, ngram=2):
		params = defaultdict (lambda: {
			'count': 0, 
			'single': False, # to distinguish with combination of words
			ngram: defaultdict (lambda: {'count': 0}, {})}, {})

		for sent in self.corpus:
			sent = self.preprocess (sent, ngram)
			params = self.count (sent, params, ngram)
		params = self.estimate_logp (params, ngram)

		return params

	def estimate_logp (self, params, ngram=2):
		for w, w_v in params.items ():
			if w_v['single'] is True:
				for history, his_v in w_v[ngram].items ():
					join_count = his_v['count']
					his_count = params[history]['count']
					p = join_count / his_count
					his_v['logp'] = math.log (p)
		return params			

	def log_to_p (self, logp):
		return math.exp (logp)

	def get_logp (self, params, w, history, ngram):
		return params[w][ngram][history]['logp']

	def compute_pp (self, logp, N):
		# calcualte perplexity of a N-gram
		return -1 * logp / N

	# NEEDFIX: implement it.
	def handle_unknown (self, sent, params, ngram=2):
		return params

	# NEEDFIX: implement it.	
	def smooth (self, params): pass

	def evaluate (self, test_corpus, params, ngram=2):
		# Assume test_corpus has been preprocess to add the start-sentence and end-sentence tokens
		logp = 0
		N = 0
		for sent in test_corpus:
			params = self.handle_unknown (sent, params, ngram)
			start_index = end_index = 0
			word_num = len (sent)
			N += (word_num - (ngram - 1)) # exclude the n number of <s>
			for i in range (word_num):
				start_index = i
				end_index = start_index + ngram - 1
				if end_index >= word_num: break		
				w = sent [end_index]
				history = ' '.join (sent [start_index:end_index])
				logpi = self.get_logp (params, w, history, ngram)
				logp += logpi
		pp = self.compute_pp (logp, N)
		return pp

if __name__ == '__main__':
	from nltk import word_tokenize
	data_dir =  '../mt/source_data/HLTNAACL_2003/fr-en/English-French.training/English-French/training/'
	target_file = data_dir + 'hansard.36.1.house.debates.00{}.e'
	tcorpus = []

	filenumber = 1

	for i in range (filenumber):
		with open (target_file.format (i+1)) as efd:
			tcorpus.extend (word_tokenize (j) for j in efd)


	print ('--- Bigram ---')
	ngram = 2	
	m = Ngram (tcorpus)
	params = m.train (ngram)
	pp = m.evaluate (tcorpus, params, ngram)
	print (pp)

	print ('--- Trigram ---')
	ngram = 3
	m = Ngram (tcorpus)
	params = m.train (ngram)
	pp = m.evaluate (tcorpus, params, ngram)
	print (pp)

	print ('--- Trigram ---')
	ngram = 4
	m = Ngram (tcorpus)
	params = m.train (ngram)
	pp = m.evaluate (tcorpus, params, ngram)
	print (pp)	