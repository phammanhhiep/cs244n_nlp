import os, sys
sys.path.insert (0, os.getcwd ())
import hmm_tagger, decoder
from collections import defaultdict
from itertools import product

HMM_Tagger = hmm_tagger.HMM_Tagger
Preprocessing = hmm_tagger.Preprocessing
Decoder = decoder.Decoder

class HMM_Tagger_2 (HMM_Tagger):
	# Bi/Trigram tagger
	# Smoothing using deleted interpolation
	# Handle unknown words
	# Adding more features.

	@staticmethod	
	def _estimate_prob_uni (params):
		# estimate transition probabilities and unigram probabilities
		for s, v in params['A'].items ():
			v['prob'] = v['count'] / params['_stat_']['token']['count']
		return params

	@classmethod	
	def _estimate_prob (cls, params): 
		params = cls._estimate_prob_A (params)
		params = cls._estimate_prob_B (params)
		params = cls._estimate_prob_uni (params)
		return params

	@staticmethod	
	def estimate_lambda_set (params, ngram=2):
		# Deleted interpolation algorithm
		lambda_set = [0.1, 0.4, 0.5] if ngram == 3 else [0.3, 0.7]
		return lambda_set

	@classmethod	
	def decode (cls, params, O, startsymbol='<s>', endsymbol='</s>', ngram=2):
		# Find the most probable tag sequence, given a word sequence
		processed_O = cls.preprocess_obs (O, startsymbol, endsymbol, ngram)
		params = cls.handle_unknown_words (params, processed_O, ngram)
		# params = cls.smooth (params, processed_O, ngram=ngram)
		best_path, p = Decoder.decode (params, processed_O, ngram=ngram)
		return best_path, p

	@staticmethod
	def	gen_combined_t_primes (params, O, i, ngram=2):
		# i: index of the current word
		t_primes = []
		nw = O[i-ngram+1:i] # a list of (ngram - 1) previous words of the word in question
		nw_num = len (nw)
		for j in range (nw_num):
			temp_t = list (params['B'][nw[j]]['cond'].keys ())
			t_primes.append (temp_t)
		combined_t = [i for i in product (*t_primes)]
		return combined_t		

	@classmethod
	def smooth (cls, params, O, ngram=2):
		# assume lambda_set elements are ordered from Uni-gram to higher N-gram  
		# t_primes elements are ordered from Uni-gram to higher N-gram 
		# assume no unknown word. after smooth, always return a non-zero value
		# smooth probability value of tags of words with N-gram dependency if the probability is zero
		lambda_set = cls.estimate_lambda_set (params, ngram)
		wnum = len (O)
		for j in range (wnum):
			w = O[j]
			if j < ngram - 1: continue
			all_t_primes = cls.gen_combined_t_primes (params, O, j, ngram)
			for t in params['B'][w]['cond']:
				for t_primes in all_t_primes:
					t_primes_str = ' '.join (t_primes)
					if len (t_primes) == (ngram - 1) and params['A'][t]['cond'][t_primes_str]['prob'] == 0:
						temp_p = 0
						t_primes_list = [t] # including all n-grams	
						n = len (t_primes)				
						for i in range (n):
							temp_t = t_primes[n-i-1:n]
							temp_t_str = ' '.join (temp_t)
							t_primes_list.append (temp_t_str)
						n = len (t_primes_list)
						for i in range (n):
							tpi = t_primes_list[i]
							li = lambda_set[i]
							ngram_p = params['A'][t]['cond'][tpi]['prob'] if i > 0 else params['A'][t]['prob']
							temp_p += li * ngram_p
						params['A'][t]['cond'][t_primes_list[-1]]['prob'] = temp_p
		return params

	@staticmethod
	def handle_unknown_words (params, O, ngram=2):
		return params

if __name__ == '__main__':
	train_sents = []; train_tags = []
	start = 1
	file_num = 10
	for i in range (start, start + file_num):
		fname = 'source_data/brown/ca0{}' if i <= 9 else 'source_data/brown/ca{}'
		fname = fname.format (i)
		with open (fname) as fd:
			for sent in fd:
				if len (sent) > 1:
					temp_sent, temp_tags = Preprocessing.prepare (sent)
					if len (temp_sent) and len (temp_tags):
						train_sents.append (temp_sent); train_tags.append (temp_tags)
	
	test_sents = []; test_tags = []		
	start = 40
	file_num = 1
	for i in range (start, start + file_num):
		fname = 'source_data/brown/ca0{}' if i <= 9 else 'source_data/brown/ca{}'
		fname = fname.format (i)
		with open (fname) as fd:
			for sent in fd:
				if len (sent) > 1:
					temp_sent, temp_tags = Preprocessing.prepare (sent)
					if len (temp_sent) and len (temp_tags):
						test_sents.append (temp_sent); test_tags.append (temp_tags)

	ngram = 3
	default_prob = 1/100000			
	params = HMM_Tagger_2.train (train_sents, train_tags, ngram=ngram, default_prob=default_prob)
	performance = HMM_Tagger_2.evaluate (test_sents[:15], test_tags[:15], params, ngram=ngram)
	print (performance)