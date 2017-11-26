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

	@classmethod	
	def decode (cls, params, O, suffix_params, theta, lambda_set, max_length=10, startsymbol='<s>', endsymbol='</s>', ngram=2):
		# Find the most probable tag sequence, given a word sequence
		processed_O = cls.preprocess_obs (O, startsymbol, endsymbol, ngram)
		params = cls.handle_unknown_words (params, processed_O, suffix_params, theta, max_length, startsymbol, endsymbol)
		params = cls.smooth_transition (params, processed_O, lambda_set, ngram=ngram)
		best_path, p = Decoder.decode (params, processed_O, ngram=ngram)
		print ('best path p', p)
		return best_path, p

	@classmethod	
	def evaluate (cls, sentsq, tagsq, params, suffix_params, theta, lambda_set, max_length=10, ngram=2):	
		total = 0
		correct = 0
		snum = len (sentsq)
		for i in range (snum):
			sent = sentsq[i]
			tags = tagsq[i]
			tags_hat, p = cls.decode (params, sent, suffix_params, theta, lambda_set, max_length=max_length, ngram=ngram)
			temp_total, temp_correct = cls._is_correct_tags (tags_hat, tags)
			print ('individual performance:', temp_correct/temp_total)
			total += temp_total; correct += temp_correct
		return correct / total

	@staticmethod
	def	gen_t_primes (params, O, i, ngram=2):
		# i: index of the current word
		# geneate all possible (N-gram - 1) of t_primes given the current word and its ngram-1 preceding words
		t_primes = []
		words = O[i-ngram+1:i] # a list of (ngram - 1) previous words of the word in question
		word_num = len (words)
		for j in range (word_num):
			cur_w = words[j]
			temp_t = list (params['B'][cur_w]['cond'].keys ())
			t_primes.append (temp_t)
		combined_t = [i for i in product (*t_primes)]
		return combined_t		
	
	@staticmethod	
	def estimate_lambda_set (params, ngram=2):
		# Deleted interpolation algorithm
		lambda_set = [0] * ngram
		for t,v in params['A'].items ():
			for tp, vp in v['cond'].items ():
				if len (tp.split ()) == (ngram - 1) and vp['count'] > 0:
					# STOP here. should backoff.
					f = []
					for i in range (ngram-1):
						no_tp = ' '.join (tp.split (' ')[i:])
						no = v['cond'][no_tp]['count'] - 1
						if i == (ngram - 2): de = params['A'][no_tp]['count'] - 1
						else:
							temp_tp = tp.split (' ')
							temp_tp_1 = ' '.join (temp_tp[-1:])
							temp_tp_2 = ' '.join (temp_tp[:-1])
							de = params['A'][temp_tp_1]['cond'][temp_tp_2]['count'] - 1
						if de == 0: f.insert (0, 0)
						else: f.insert (0, no / de) # freqencies are estimated in deceding order of ngram
					# calculate ratio between t and token count
					no = v['count'] - 1
					de = params['_stat_']['token']['count'] - 1
					if de == 0: f.insert (0, 0)
					else: f.insert (0, no / de)
					# increase lambda with max
					j = max (f)
					li = f.index (j)
					lambda_set[li] += j
		s = sum (lambda_set)
		lambda_set = [l / s for l in lambda_set]
		return lambda_set
	
	@classmethod
	def smooth_transition (cls, params, O, lambda_set, ngram=2):
		# assume lambda_set elements are ordered from Uni-gram to higher N-gram  
		# t_primes elements are ordered from Uni-gram to higher N-gram 
		# assume no unknown word. after smooth, always return a non-zero value
		# smooth probability value of tags of words using interpolation with lower N-grams if the trigram probability is zero
		wnum = len (O)
		for j in range (wnum):
			w = O[j]
			if j < ngram - 1: continue # not consider start symbols
			t_primes = cls.gen_t_primes (params, O, j, ngram)
			for t in params['B'][w]['cond']:
				if len (t.split (' ')) > 1: continue # just make sure only single tags are considered
				for tp in t_primes:
					n = len (tp)
					tp_str = ' '.join (tp)
					if n == (ngram - 1) and params['A'][t]['cond'][tp_str]['prob'] == 0:
						smooth_list = [t] # including all n-grams					
						for i in range (n):
							temp_tp = tp[n-i-1:n] # take smaller n-gram first  
							temp_tp_str = ' '.join (temp_tp)
							smooth_list.append (temp_tp_str)
						snum = len (smooth_list)
						temp_p = 0
						for i in range (snum):
							tpi = smooth_list[i]
							li = lambda_set[i]
							p = params['A'][t]['cond'][tpi]['prob'] if i > 0 else params['A'][t]['prob']
							temp_p += li * p
						target_tp = smooth_list[-1]
						params['A'][t]['cond'][target_tp]['prob'] = temp_p
		return params

	@classmethod	
	def handle_unknown_words (cls, params, O, suffix_params, theta, max_length=10, startsymbol='<s>', endsymbol='</s>'):
		params = UNK_Handler.handle_unknown_words (params, O, suffix_params, theta, max_length, startsymbol, endsymbol)
		return params	

class UNK_Handler:
	# guess all possible tags for words, and estimate their probability.
	@classmethod
	def handle_unknown_words (cls, params, O, suffix_params, theta, max_length=10, startsymbol='<s>', endsymbol='</s>'):
		for Oi in O:
			if len (params['B'][Oi]['cond'].keys ()) > 0 and params['B'][Oi]['count'] > 0: continue
			s = cls.get_selected_suffix (Oi, suffix_params, max_length)
			suffix_params = cls.smooth (s, theta, suffix_params)
			params = cls.estimate_B (Oi, s, params, suffix_params)
		return params

	@classmethod	
	def train (cls, params, max_length=10, startsymbol='<s>', endsymbol='</s>'):
		suffix_params = cls.count (params, max_length, startsymbol, endsymbol)
		suffix_params = cls._train (suffix_params)
		theta = cls.estimate_theta (suffix_params)
		return suffix_params, theta	

	@staticmethod
	def _train (suffix_params):
		for s,v in suffix_params['S'].items ():
			v['prob'] = v['count'] / suffix_params['_stat_']['S']['count']
			for t,vt in v['cond'].items ():
				vt['prob'] = vt['count'] / v['count'] if v['count'] > 0 else 0
		for t,v in suffix_params['T'].items ():	
			v['prob'] = v['count'] / suffix_params['_stat_']['T']['count']
		return suffix_params

	@staticmethod
	def count (params, max_length, startsymbol, endsymbol):
		# count all suffix of all data. Many other variants only consider infrequent words.
		# With small dataset, the approach is reasonable
		suffix_params = {
			'S': defaultdict (lambda: {'cond': defaultdict (lambda: {'count': 0, 'prob': 0}, {}), 'count': 0, 'prob': 0}, {}),
			'T': defaultdict (lambda: {'count': 0, 'prob': 0}, {}),
			'_stat_': {'S': {'count': 0}, 'T': {'count': 0}}
		}

		for w,v in params['B'].items ():
			if w in [startsymbol, endsymbol]: continue
			if v['count'] > 10: continue # should be an parameter when calling
			wlen = len (w)
			maxi = max_length if max_length < wlen else wlen
			for i in range (maxi):
				suffix = w[wlen-i-1:wlen]
				suffix_params['S'][suffix]['count'] += v['count']
				suffix_params['_stat_']['S']['count'] += v['count']
				for t,vt in v['cond'].items ():
					suffix_params['S'][suffix]['cond'][t]['count'] += vt['count']
		for w,v in params['B'].items ():
			for t,vt in v['cond'].items ():					
				suffix_params['T'][t]['count'] += vt['count']
				suffix_params['_stat_']['T']['count'] += vt['count']
		return suffix_params

	@staticmethod
	def estimate_theta (suffix_params):
		tnum = len (suffix_params['T'].keys ())
		total_p = sum ([suffix_params['T'][t]['prob'] for t in suffix_params['T']])
		ave_p = total_p / tnum
		sd = sum ([pow(suffix_params['T'][t]['prob'] - ave_p, 2) for t in suffix_params['T']]) / (tnum - 1)
		return sd

	@staticmethod
	def get_selected_suffix (Oi, suffix_params, max_length):
		maxi = max_length if max_length < len (Oi) else len (Oi)
		return Oi[-maxi:]

	@classmethod
	def smooth (cls, s, theta, suffix_params):
		# if a suffix does not exit, backoff to suffix with shorter length.
		# smooth posterior probability of a suffix, and conditional probability of a tag given a suffix
		def _backoff (sp, s):
			ori_s = s
			if len (sp['S'][s]['cond']) == 0: 
				while s != s[-len (s) + 1:]:
					s = s[-len (s) + 1:]
					if len (sp['S'][s]['cond']) != 0:
						sp['S'][ori_s]['prob'] = sp['S'][s]['prob']
						break
			return s
		ori_s = s
		s = _backoff (suffix_params, s)
		for t,vt in suffix_params['S'][s]['cond'].items ():
			p = cls.smooth_func (s, t, theta, suffix_params)
			suffix_params['S'][ori_s]['cond'][t]['prob'] = p
		return suffix_params

	@classmethod
	def smooth_func (cls, s, t, theta, suffix_params):
		less_suffix = s[-len (s)+1:]
		if less_suffix != s:
			p = (suffix_params['S'][s]['cond'][t]['prob'] + theta * cls.smooth_func (less_suffix, t, theta, suffix_params)) / (1 + theta)
		else:
			p = (suffix_params['S'][s]['cond'][t]['prob'] + theta * suffix_params['T'][t]['prob']) / (1 + theta)
		return p

	@staticmethod
	def estimate_B (Oi, s, params, suffix_params):
		# convert from P(t|s) to P(s|t) with s is suffix and t is tag
		for t,vt in suffix_params['S'][s]['cond'].items ():
			params['B'][Oi]['cond'][t]['prob'] = vt['prob'] * suffix_params['S'][s]['prob'] / suffix_params['T'][t]['prob']
		return params
	
if __name__ == '__main__':
	train_sents = []; train_tags = []
	start = 1
	file_num = 20
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
	start = 41
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
	max_length = 10
	params = HMM_Tagger_2.train (train_sents, train_tags, ngram=ngram)
	suffix_params, theta = UNK_Handler.train (params, max_length)
	lambda_set = HMM_Tagger_2.estimate_lambda_set (params, ngram=ngram)
	performance = HMM_Tagger_2.evaluate (test_sents[:5], test_tags[:5], params, suffix_params, theta, lambda_set, max_length=max_length, ngram=ngram)
	print ('Total performance:', performance)