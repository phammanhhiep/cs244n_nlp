import random, math
from collections import defaultdict

class Topic_Classifier:
	# Topic classification using Naive Bayes classifier
	# Estimate parameters using MLE
	# Multi-label classification - classify an item to more than one topic

	@classmethod
	def train (cls, docs):
		params = cls.count_params (docs)
		params = cls.estimate_params (params)
		return params

	@staticmethod	
	def count_params (docs):
		params = {
			'C': defaultdict (lambda: {'d_count': 0, 'f_count': 0, 'notc_d_count': 0, 'notc_f_count': 0, 'features': defaultdict(lambda: {'count': 0, 'notc_count': 0, 'log': 0, 'notc_log': 0}, {})}, {}),
			'_stat_': {'doc': {'count': 0}, 'V': {'count': 0}}
		}
		V = []

		for d in docs:
			ws = d[0]
			cs = d[1]
			params['_stat_']['doc']['count'] += 1
			for c in cs:
				params['C'][c]['d_count'] += 1
				for w in ws:
					V.append (w)
					params['C'][c]['f_count'] += 1
					params['C'][c]['features'][w]['count'] += 1
			V = list (set (V))
		params['_stat_']['V']['count'] = len (V)

		for c,cv in params['C'].items ():
			cl = [i for i in params['C'].keys () if i != c]
			for d in docs:
				ws = d[0]
				cs = d[1]
				exist = sum([i in cs for i in cl])
				if exist > 0:
					params['C'][c]['notc_d_count'] += 1
					for w in ws:
						V.append (w)
						params['C'][c]['notc_f_count'] += 1
						params['C'][c]['features'][w]['notc_count'] += 1							
		return params			

	@staticmethod
	def estimate_params (params):
		# Add-one (Laplace) smoothing
		for c, vc in params['C'].items ():
			vc['log'] = math.log (vc['d_count'] / params['_stat_']['doc']['count'])
			vc['notc_log'] = math.log (vc['notc_d_count'] / params['_stat_']['doc']['count'])
			for f,vf in vc['features'].items ():
				vf['log'] = math.log((vf['count'] + 1) / (vc['f_count'] + params['_stat_']['V']['count']))
				vf['notc_log'] = math.log((vf['notc_count'] + 1) / (vc['notc_f_count'] + params['_stat_']['V']['count']))
		return params	

	@staticmethod	
	def handle_unknown (ws, params):
		# do nothing. with default dict, log of unknown word is set as 0 in default.
		return params
	
	@classmethod
	def classify (cls, docs, params):
		labels = []
		for ds in docs:
			ws = ds[0]
			c_list = []
			params = cls.handle_unknown (ws, params)	
			for c,cv in params['C'].items ():
				c_log, notc_log = cls.estimate_log_c (ws, cv)
				if c_log >= notc_log: # consider if to label when the two values equal
					c_list.append (c)
			labels.append (c_list)
		return labels

	@staticmethod			
	def estimate_log_c (ws, cparam):
		# estimate log likelihood of a class given a document
		# ws: a word sequence; cparam: parameters of a classifier
		# return both log p of a class and log p of not-that-class
		c_log = cparam['log']
		notc_log = cparam['notc_log']
		for w in ws:
			c_log += cparam['features'][w]['log']
			notc_log += cparam['features'][w]['notc_log']
		return c_log, notc_log

	def evaluate (topics, pred_topics):
		accuracy = recall = precise = F = 0
		dnum = len (topics) 
		if dnum != len (pred_topics): raise Exception ('Not match document length')
		for i in range (dnum):
			pass
			# STOP here. Need to reasearch about measure metrics	

	def cv (): pass


def preprocess (content):
	new_content = []
	topic = []
	blist = ['\\n', '\\t']
	for sent in content:
		for i in blist:
			sent = sent.replace (i, '')
		sent = sent.strip ()
		c = sent[:sent.index ('<')-1].strip ().split (' ')
		t = sent[sent.index ('<') + 1:-1].strip ().split ('-')
		n = [c,t]
		new_content.append (n)
	return new_content

def random_pick (content, num):
	tr_num = len (content)
	tr_num = math.floor(tr_num * num)
	train = random.sample (content, tr_num)
	test = [i for i in content if i not in train]
	return train, test

if __name__ == '__main__': 
	fname = 'source_data/gs_conversation'
	docs = []
	with open (fname) as fd:
		docs = [i for i in fd]
	new_docs = preprocess (docs)
	trainset, testset = random_pick (new_docs, 0.8)

	params = Topic_Classifier.train (trainset)
	labels = Topic_Classifier.classify (testset, params)

	for i in range (len (labels)):
		print (testset[i], labels[i])

