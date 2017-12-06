import numpy as np
import math, time, random
from itertools import product, permutations

class Hierarchical:
	def __init__ (self, distance_metric, m):
		# m: number of clusters
		self.distance_metric = distance_metric
		self.m = m

	def measure_distance (self, cp, jp, wp, V, C):
		return self.distance_metric.measure (cp, jp, wp, V, C)
	
	def cluster (self, docs, V, C):
		time_start = Toolkit.timeit_start ()
		m = self.m
		clusters = [[] for i in range (m)]	
		count = self.count (docs, V, C)
		print ('- done count: {} s'.format (Toolkit.timeit (time_start)));time_start = Toolkit.timeit_start ()		
		cp, jp, wp, cpp = self.estimate_p (count)
		print ('- done estimate_p: {} s'.format (Toolkit.timeit (time_start)));time_start = Toolkit.timeit_start ()			
		distance = self.measure_distance (cp, jp, wp, V, C)
		print ('- done measure_distance: {} s'.format (Toolkit.timeit (time_start)));time_start = Toolkit.timeit_start ()		
		ranking = self.rank (cp, wp, cpp, V, C)
		print ('- done rank: {} s'.format (Toolkit.timeit (time_start)));time_start = Toolkit.timeit_start ()		
		init_index = ranking[:m]
		clusters = self.init_clusters (clusters, init_index)
		for i in ranking[m:]:
			L = merged_cli = empty_cli = None # init
			clusters, L, merged_cli, empty_cli = self.create_cluster (clusters, i, distance, L, merged_cli, empty_cli)
		print ('- done create_cluster: {} s'.format (Toolkit.timeit (time_start)))
		return clusters	

	def init_clusters (self, clusters, index):
		# index: index of a word in the vocabulary
		clnum = len (clusters)
		for i in range (clnum):
			clusters[i].append (index[i])
		return clusters
		
	def create_cluster (self, clusters, i, distance, L, merged_cli, empty_cli):
		# empty_cli: index of a cluster
		# i: index of a word in vocabulary
		# L: a matrix of linkage values with size len (clusters) x len (clusters)
		clusters, L, merged_cli, empty_cli = self.merge (clusters, distance, L, merged_cli, empty_cli)
		clusters[empty_cli].append (i)
		return clusters, L, merged_cli, empty_cli 

	def merge (self, clusters, distance, L, merged_cli, empty_cli): 
		# merge two most similar clusters
		clnum = len (clusters)
		cindex = [i for i in range (clnum)]
		if empty_cli is None and merged_cli is None and L is None: # init
			pairs = list (permutations (cindex, 2))
			L = [[None] * clnum for i in range(clnum)] 
		else:
			remained_cindex = cindex[:merged_cli] + cindex[merged_cli:]
			remained_cindex = cindex[:empty_cli] + cindex[empty_cli:]
			pairs = list (product ([merged_cli, empty_cli], remained_cindex))
			pairs.append ((merged_cli, empty_cli))
		for p in pairs:
			if p[0] > p[1]: continue # only consider cells of upper triangular
			cl1 = clusters[p[0]]
			cl2 = clusters[p[1]]
			l = self.measure_linkage (cl1, cl2, distance)
			row = p[0]; col = p[1]
			L[row][col] = l

		flatten_L = [i for l in L for i in l]
		L_sum = sum ([i for l in L for i in l if i is not None])
		flatten_L = [i if i is not None else L_sum for i in flatten_L]
		maxl = min (flatten_L)
		maxl_index = flatten_L.index (maxl)
		merged_cli = math.floor (maxl_index / clnum)
		empty_cli = maxl_index - merged_cli * clnum
		clusters[merged_cli].extend (clusters[empty_cli])
		clusters[empty_cli] = []
		return clusters, L, merged_cli, empty_cli 

	def measure_linkage (self, cl1, cl2, distance):
		# Average linkage
		linkage = 0
		pd = list (product (cl1, cl2))
		numpd = len (pd)
		for i in pd:
			i = sorted (i)
			linkage += distance[i[0]][i[1]]
		return linkage / numpd

	def rank (self, cp, wp, cpp, V, C):
		# rank words according to MI to the class distribution
		# rank in desceding order
		mil = []
		vnum = len (V)
		cnum = len (C)
		ranking = []
		for i in range (vnum):
			mi = 0
			w_p = wp[i]
			for j in range (cnum):
				c_p = cpp[j] 
				cond_p = cp[j][i]
				join_p = cond_p * w_p
				mi += (join_p * math.log (join_p / (w_p * c_p), 2) if join_p != 0 else 0)
			mil.append (mi)
		ranking = list (range (vnum))
		ranking = sorted (ranking, key=lambda x: mil[x], reverse=True)
		return ranking

	def extract_metadata (self, docs):
		V = []; C = []
		for content in docs:
			w,c = content
			C.extend (c); V.extend (w)
			C = list (set (C)); V = list (set (V))
		return V,C

	def count (self, docs, V, C):
		# could frequency of word in each class
		vlen = len (V)
		clen = len (C)		
		cm = [[0] * vlen for i in range (clen)]
		for d in docs:
			ws = d[0]
			c = d[1]
			for w in ws:
				wi = V.index (w)
				for cj in c:
					ci = C.index (cj)
					cm[ci][wi] += 1
		return cm

	def estimate_p (self, count): 
		cp = self.estimate_cp (count)
		wp = self.estimate_wp (count)
		jp = self.estimate_jp (count)
		cpp = self.estimate_cpp (count)
		return cp,jp,wp,cpp

	def estimate_cpp (self, count):
		# estimate priori probability of each class
		total = np.sum (count)
		ccount = np.sum (count, axis=1)
		return ccount / total

	def estimate_cp (self, count):
		# estimate conditional probability of each class given a word
		# add one to smooth each unseen word in a class
		count = np.array (count)
		wcount = count.sum (axis=0)
		return (count + 1) / (wcount + 1) 

	def estimate_wp (self, count):
		# estimate priori probability of each word in the vocabulary
		total = np.sum (count)
		wcount = np.sum (count, axis=0)
		return wcount / total

	def estimate_jp (self, count):
		# estimate the conditional probability of each class given a pair of words
		# add one to smooth unseen pair of words in a class
		comb_count = []
		jp = []
		for cc in count:
			m = []
			vnum = len (cc)
			for i in range (vnum):
				wc1 = cc[i]
				n = [1] * (i+1) # first values does not matter. Not used. only cells in upper triangle of the matrix are used. 
				for wc2 in cc[i+1:]:
					n.append (wc1 + wc2 + 1) # smoothing
				m.append (n)
			comb_count.append (m)
		total = np.sum (comb_count, axis=0) # total count of each pair with all classes. a matrix with same size as each matrix m
		for cc in comb_count:
			jp.append (np.array (cc) / total)
		return jp

class KL2Mean:
	@classmethod
	def measure (cls, cp, jp, wp, V, C):
		# cp: conditional probability of each class given a word
		# jp: conditional probability of each class given a pair of words
		# wp: priori probability of each word in the vocabulary 
		cnum = len (C)
		vnum = len (V)
		distance = []
		for i in range (vnum):
			n = [0] * (i+1)	# the value of zero does not matter. only cells in upper triangular part are used.	
			for j in range (i+1, vnum):
				KL = 0
				wi_p = wp[i]
				wj_p = wp[j]				
				for z in range (cnum):
					cwi_p = cp[z][i]
					cwj_p = cp[z][j]
					cwji_p = jp[z][i][j] 
					KL += cls.measure_KL (cwi_p, cwj_p, cwji_p, wi_p, wj_p)
				n.append (KL)
			distance.append (n)
		return distance

	@staticmethod
	def measure_KL (cwi_p, cwj_p, cwji_p, wi_p, wj_p):
		KLi = wi_p * cwi_p * math.log (cwi_p/cwji_p, 2)
		KLj = wj_p * cwj_p * math.log (cwj_p/cwji_p, 2)
		return KLi + KLj

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

	trainset, testset = Toolkit.random_pick (docs, 0.5)
	trainset = trainset[:20]

	h = Hierarchical (KL2Mean, 5)
	V,C = h.extract_metadata (trainset)
	clusters = h.cluster (trainset, V, C)
	print ([len (i) for i in clusters])
			