from collections import defaultdict
import random, math 

class Ngram:
	# Estimate probability of N-gram with MLE
	# Support unigram, bigram, and trigram
	def __init__ (self, corpus=None):
		self.corpus = corpus

	def preprocess (self, sent, ngram=2):
		# add <s> and </s> at the begining and end of a sentence respectively
		new_sent = [i for i in sent]
		for i in range (ngram - 1):
			new_sent.insert (0, '<s>')
		new_sent.append ('</s>')
		return new_sent

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

	def precount (self, sent, count):
		word_num = len (sent)
		for i in range (word_num):
			w = sent [i]
			count[w]['count'] += 1
		return count		

	def handle_train_unknown (self, sent, count, unk_threshold):
		# handle both unknown word
		unknown_word = '<UNK>'
		new_sent = []
		for w in sent:
			if count[w]['count'] <= unk_threshold: w = unknown_word
			new_sent.append (w)
		return new_sent

	def train (self, ngram=2, unk_threshold=1):
		def _gen_word_dict (ngram):
			w = {
				'count': 0,
				'single': False, # to distinguish with combination of words
			}
			w.update ({ngram-k: defaultdict (lambda: {'count': 0, 'logp': 0}, {}) for k in range (ngram-1)})
			return w
		params = defaultdict (lambda: _gen_word_dict (ngram), {})

		oricounts = defaultdict (lambda: {'count': 0}, {})

		for sent in self.corpus:
			oricounts = self.precount (sent, oricounts)
		for sent in self.corpus:
			sent = self.handle_train_unknown (sent, oricounts, unk_threshold)
			sent = self.preprocess (sent, ngram)
			params = self.count (sent, params, ngram)
		params = self.estimate_logp (params, ngram)
		return params
	
	def smooth (self, params):
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

	def handle_test_unknown (self, sent, params):
		# handle both unknown word
		unknown_word = '<UNK>'
		new_sent = []
		for w in sent:
			if params[w]['count'] == 0: w = unknown_word
			new_sent.append (w)
		return new_sent		

	def evaluate (self, test_corpus, params, ngram=2):
		# Assume test_corpus has been preprocess to add the start-sentence and end-sentence tokens
		logp = 0
		N = 0
		for sent in test_corpus:
			sent = self.preprocess (sent)
			sent = self.handle_test_unknown (sent, params)
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

	print ('--- Bigram ---')
	ngram = 2	
	m = Ngram (tcorpus)
	params = m.train (ngram)
	pp = m.evaluate (trial_tcorpus, params, ngram)
	print (pp)
	c=0
	for k,v in params.items ():
		print (k,v)
		c += 1
		if c == 1: break

	print ('--- Trigram ---')
	ngram = 3
	m = Ngram (tcorpus)
	params = m.train (ngram)
	pp = m.evaluate (trial_tcorpus, params, ngram)
	print (pp)
	c=0
	for k,v in params.items ():
		print (k,v)
		c += 1
		if c == 1: break

	print ('--- Trigram ---')
	ngram = 4
	m = Ngram (tcorpus)
	params = m.train (ngram)
	pp = m.evaluate (trial_tcorpus, params, ngram)
	print (pp)
	c=0
	for k,v in params.items ():
		print (k,v)
		c += 1
		if c == 1: break