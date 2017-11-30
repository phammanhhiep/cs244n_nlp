import random, math, time
from collections import defaultdict

class NBClassifier:
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

	@classmethod	
	def evaluate (cls, Y, Z):
		# Y: actual topics; Z: predicted topics
		# example-based.
		dnum = len (Y) 
		if dnum != len (Z): raise Exception ('Not match document length')
		total_YZ = total_Y = total_Z = match_YZ = 0
		for i in range (dnum):
			Yi = Y[i]
			Zi = Z[i]
			total_YZ += len (Yi) + len (Zi)
			total_Y += len (Yi)
			total_Z += len (Zi)
			match_YZ += len (set (Yi).intersection (set (Zi)))
		return cls.compute_metric (dnum, total_YZ, total_Y, total_Z, match_YZ)

	@staticmethod	
	def compute_metric (n, total_YZ, total_Y, total_Z, match_YZ):
		accuracy = (match_YZ / total_YZ) / n
		precision = (match_YZ / total_Z) / n if total_Z > 0 else 'N/A'
		recall = (match_YZ / total_Y) / n
		F1 = 2 * (match_YZ / total_YZ) / n
		return accuracy, precision, recall, F1

	def cv (): pass

class Toolkit:
	@staticmethod
	def random_pick (content, num):
		tr_num = len (content)
		tr_num = math.floor(tr_num * num)
		train = random.sample (content, tr_num)
		test = [i for i in content if i not in train]
		return train, test

	@staticmethod
	def timeit_start ():
		return time.time()

	@staticmethod
	def timeit (start):
		return time.time () - start

if __name__ == '__main__': 
	time_start = Toolkit.timeit_start ()
	from bs4 import BeautifulSoup as BS
	from nltk import word_tokenize
	ori_docs = ''
	filename = 'source_data/reuters21578/reut2-{}.sgm'
	filenum = 0
	filenum_prefix = '00{}' if filenum <= 9 else '0{}'
	filenum = filenum_prefix.format (filenum)
	filename = filename.format (filenum)
	with open (filename) as fd:
		ori_docs = fd.read ()
	ori_docs = BS (ori_docs, 'lxml')
	docs = []
	for d in ori_docs.find_all ('reuters'):
		if d['topics'] == 'YES':
			topic = [i.text for i in d.find ('topics').find_all ('d')]
			content = d.find('text')
			if content.find('body'):
				content = content.find('body').text
			else:
				if content.find ('author'): content.find ('author').decompose ()
				if content.find ('title'): content.find ('title').decompose ()
				if content.find ('dateline'): content.find ('dateline').decompose ()
				content = content.text
			content = word_tokenize (content)
			docs.append ([content, topic])

	print ('- done preprocess: {} s'.format (Toolkit.timeit (time_start)))		
	time_start = Toolkit.timeit_start ()

	trainset, testset = Toolkit.random_pick (docs, 0.9)
	params = NBClassifier.train (trainset[:])

	print ('- done train: {} s'.format (Toolkit.timeit (time_start)))		
	time_start = Toolkit.timeit_start ()	

	testset = testset[:]
	predicted_labels = NBClassifier.classify (testset, params)

	print ('- done classify: {} s'.format (Toolkit.timeit (time_start)))		
	time_start = Toolkit.timeit_start ()

	labels = [t[1] for t in testset]
	accuracy, precision, recall, F1 = NBClassifier.evaluate (labels, predicted_labels)
	print ('- done evaluate: {} s'.format (Toolkit.timeit (time_start)))
	print ('accuracy: {}, precision: {}, recall: {}, F1: {}'.format (accuracy, precision, recall, F1))	

