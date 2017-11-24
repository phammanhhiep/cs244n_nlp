# Given a training data, learn parameters of a HMM with maximum likelihood estimate
# Apply HMM decoding to find the most probable sequence of tags
# Built a mechanism to add more features.

from nltk import sent_tokenize, word_tokenize
from collections import defaultdict
import os, sys
sys.path.insert (0, os.getcwd ())
import decoder
Decoder = decoder.Decoder

class HMM_Tagger:
	# Bigram tagger
	# No smoothing. No handle unknown words. Default p for unseen bigram or unknown words.
	@staticmethod
	def gen_params (default_prob=0):
		# FIX: insert a default probability to deal with unknow word. Need a better solutions.
		return {
			'A': defaultdict (lambda: {'cond': defaultdict (lambda: {'count': 0, 'prob': default_prob}, {}), 'count': 0, 'prob': default_prob}, {}),
			'B': defaultdict (lambda: {'cond': defaultdict (lambda: {'count': 0, 'prob': default_prob}, {}), 'count': 0}, {}),
			'_stat_': {'token': {'count': 0}}
		}	

	@staticmethod
	def preprocess_obs (sq, startsymbol, endsymbol, ngram):
		sq = [wi for wi in sq]
		for k in range (ngram-1): sq.insert (0, startsymbol)
		sq.append (endsymbol)
		return sq

	@staticmethod
	def preprocess_states (sq, startsymbol, endsymbol, ngram):
		sq = [wi for wi in sq]
		for k in range (ngram-1): sq.insert (0, startsymbol)
		sq.append (endsymbol)
		return sq				

	@classmethod
	def train (cls, sents, tagsq, ngram=2, default_prob=0):
		# ws: word sequence; ts: tag sequence
		# return transition probabilities P(s(t) | s(t-1)) and emission probabilities P (wi | s(t))
		params = cls.gen_params (default_prob)
		params = cls._count (sents, tagsq, params, ngram=ngram)
		params = cls._estimate_prob (params)
		return params

	@classmethod
	def _count (cls, sents, tagsq, params, startsymbol='<s>', endsymbol='</s>', ngram=2):
		snum = len (sents)
		for i in range (snum):
			words = sents[i]; tags = tagsq[i]
			words = cls.preprocess_obs (words, startsymbol, endsymbol, ngram)
			tags = cls.preprocess_states (tags, startsymbol, endsymbol, ngram)
			wnum = len (words)
			for j in range (wnum):
				t = tags[j]
				w = words[j]
				params['A'][t]['count'] += 1
				params['B'][w]['count'] += 1
				params['_stat_']['token']['count'] += 1
				if w == startsymbol: params['B'][w]['cond'][t]['count'] += 1
				if j >= ngram - 1:
					lower_index = j-ngram+1
					for j_prime in range (lower_index, j):
						prev_t = ' '.join (tags[j_prime:j]) # assume elements are strings	
						if j - lower_index > 1: params['A'][prev_t]['count'] += 1 # count combination of tags
						params['A'][t]['cond'][prev_t]['count'] += 1
					params['B'][w]['cond'][t]['count'] += 1
		return params	

	@classmethod	
	def _estimate_prob (cls, params): 
		params = cls._estimate_prob_A (params)
		params = cls._estimate_prob_B (params)
		return params

	@staticmethod	
	def _estimate_prob_A (params):
		# estimate transition probabilities
		for s, v in params['A'].items ():
			if len (s.split (' ')) > 1: continue # not find p of N-gram
			cond = v['cond']
			for s_prime in cond.keys ():
				cond[s_prime]['prob'] = cond[s_prime]['count'] / params['A'][s_prime]['count']
		return params

	@staticmethod	
	def _estimate_prob_B (params):
		# estimate emission probabilities	
		for w, v in params['B'].items ():
			cond = v['cond']
			for s in cond.keys ():
				cond[s]['prob'] = cond[s]['count'] / params['A'][s]['count']
		return params

	@staticmethod	
	def decode (params, O, ngram=2):
		# Find the most probable tag sequence, given a word sequence
		best_path, p = Decoder.decode (params, O, ngram=ngram)
		return best_path, p

	@classmethod	
	def decode (cls, params, O, startsymbol='<s>', endsymbol='</s>', ngram=2):
		# Find the most probable tag sequence, given a word sequence
		processed_O = cls.preprocess_obs (O, startsymbol, endsymbol, ngram)
		best_path, p = Decoder.decode (params, processed_O, ngram=ngram)
		return best_path, p

	@staticmethod	
	def tag (words, tags): pass
		# assume tage 

	@classmethod
	def evaluate (cls, sentsq, tagsq, params, ngram=2):
		# Evaluate the result from decoding process.
		total = 0
		correct = 0
		snum = len (sentsq)
		for i in range (snum):
			sent = sentsq[i]
			tags = tagsq[i]
			tags_hat, p = cls.decode (params, sent, ngram=ngram)
			temp_total, temp_correct = cls._is_correct_tags (tags_hat, tags)
			total += temp_total; correct += temp_correct
		return correct / total

	@staticmethod	
	def _is_correct_tags (tags_hat, tags):
		tnum = len (tags_hat)
		incorrect = 0
		if tnum != len (tags):
			print (tags)
			print (tags_hat)
			msg = 'Predicted tag sequence has different length from original: {} != {}'.format (tnum, len (tags))
			raise Exception (msg)
		for i in range (tnum):
			if tags_hat[i] != tags[i]: incorrect += 1
		return tnum, tnum - incorrect

class Preprocessing:
	@staticmethod
	def strip (words, blist=None):
		# remove \n and \t
		blist = ['\n', '\t'] if blist is None else blist
		for i in blist:
			words = words.replace (i, '')
		words = words.strip ()
		return words

	@staticmethod
	def extract_tags (sent):
		# assume having no word "/"
		tags = []
		new_sent = []
		for w in sent:
			w = w.split ('/')
			if len (w) <= 1:
				new_sent = []
				tags = []
				break
			tags.append (w[1])
			new_sent.append (w[0])
		return new_sent, tags

	@staticmethod
	def word_tokenize (sent):
		return sent.split (' ')

	@staticmethod
	def sent_tokenize (input): pass
		
	@staticmethod
	def prepare (sent):
		sent = Preprocessing.strip (sent)
		sent = Preprocessing.word_tokenize (sent)
		sent, tags = Preprocessing.extract_tags (sent)
		return sent, tags

if __name__ == '__main__':
	train_sents = []; train_tags = []
	start = 1
	file_num = 5
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

	default_prob = 1/100000 # best performance
	ngram=2
	params = HMM_Tagger.train (train_sents, train_tags, ngram=ngram, default_prob=default_prob)
	performance = HMM_Tagger.evaluate (test_sents[:5], test_tags[:5], params, ngram=ngram)
	print (performance)

