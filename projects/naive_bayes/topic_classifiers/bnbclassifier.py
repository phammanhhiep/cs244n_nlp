import os, sys
sys.path.insert (0, os.getcwd ())
from topic_classifiers.nbclassifier import NBClassifier, Toolkit

class BNBClassifier (NBClassifier):
	def binarize (docs):
		for d in docs:
			d_content = d[0]
			d_content = list (set (d_content))
			d[0] = d_content
		return docs		

if __name__ == '__main__': 
	time_start = Toolkit.timeit_start ()
	from bs4 import BeautifulSoup as BS
	from nltk import word_tokenize
	import re

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
			p = re.compile (r'[A-Za-z]+\'[A-Za-z]+')
			print (p.findall (content))
							
			content = word_tokenize (content)
			docs.append ([content, topic])

	exit ()
	print ('- done preprocess: {} s'.format (Toolkit.timeit (time_start)))		
	time_start = Toolkit.timeit_start ()

	trainset, testset = Toolkit.random_pick (docs, 0.9)
	trainset = BNBClassifier.binarize (trainset)
	testset = BNBClassifier.binarize (testset)

	params = BNBClassifier.train (trainset[:])

	print ('- done train: {} s'.format (Toolkit.timeit (time_start)))		
	time_start = Toolkit.timeit_start ()	

	testset = testset[:]
	predicted_labels = BNBClassifier.classify (testset, params)

	print ('- done classify: {} s'.format (Toolkit.timeit (time_start)))		
	time_start = Toolkit.timeit_start ()

	labels = [t[1] for t in testset]
	accuracy, precision, recall, F1 = BNBClassifier.evaluate (labels, predicted_labels)
	print ('- done evaluate: {} s'.format (Toolkit.timeit (time_start)))
	print ('accuracy: {}, precision: {}, recall: {}, F1: {}'.format (accuracy, precision, recall, F1))
