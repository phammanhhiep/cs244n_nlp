import os,sys
sys.path.insert (0, os.getcwd ())
import pytest, math
import numpy as np
from feature_clustering.hierarchical import Hierarchical,KL2Mean

@pytest.mark.skip
def test_cluster ():
	data = [
		['this is my house','c1'],
		['it looks pretty old','c1'],
		['I am working on a NLP project', 'c2'],
		['the project involve alot of NLP technique', 'c2']
	]

	m = 10
	h = Hierarchical (distance_metric=KL2Mean,m=m)
	clusters = h.cluster (data)

@pytest.mark.skip
def test_count ():
	data = [
		[['this', 'is', 'my', 'house'], 'c1'],
		[['the', 'house', 'looks', 'pretty', 'old'], 'c1'],
		[['I', 'am', 'working', 'on', 'a', 'NLP', 'project'], 'c2'],
		[['the', 'project', 'involve', 'alot', 'of', 'NLP', 'technique'], 'c2']
	]

	m = 10
	h = Hierarchical (distance_metric=KL2Mean,m=m)
	v = ['this', 'is', 'my', 'house', 'the', 'looks', 'pretty', 'old', 'I', 'am', 'working', 'on', 'a', 'NLP', 'project', 'involve', 'alot', 'of', 'technique']
	c = ['c1', 'c2']
	count = h.count (data,v,c)
	expected_count = [
		[1,1,1,2,2,1,1,1,1,1,1,1,1,2,2,1,1,1,1]
	]
	for i in range (len (expected_count[0])):
		expected_count[0][i] == count[0][i]

@pytest.mark.skip
def test_estimate_cp ():
	count = [[1,2,3,4],[5,6,7,8]]
	h = Hierarchical (None, None)
	cp = h.estimate_cp (count)
	expected_cp = [[1/6,1/4,3/10,1/3], [5/6,3/4,7/10,2/3]]
	diff = np.array(expected_cp) - np.array(cp)
	for d in diff:
		for i in d:
			assert i == 0 

@pytest.mark.skip
def test_estimate_wp ():
	count = [[1,2,3,4],[5,6,7,8]]
	h = Hierarchical (None, None)
	wp = h.estimate_wp (count)
	expected_wp = [1/6,8/36,10/36,1/3]
	diff = np.array(expected_wp) - np.array(wp)
	for d in diff:
		assert d == 0 

@pytest.mark.skip
def test_estimate_jp ():
	count = [[1,2,3,4],[5,6,7,8]]
	h = Hierarchical (None, None)
	jp = h.estimate_jp (count)
	assert False # test later. not test but check by print. result as expected.

@pytest.mark.skip
def test_rank ():
	h = Hierarchical (None, None)
	C = ['c1', 'c2']
	V = ['x1', 'x2', 'x3']
	cp = [[1,2,3], [6,4,5]]
	wp = [2,1,3]
	cpp = [2,5]
	ranking = h.rank (cp, wp, cpp, V, C)
	assert False  # not test yet. Just confirm it works by looking at the result.

@pytest.mark.skip
def test_merge ():
	h = Hierarchical (KL2Mean, None)
	distance = [
		[0, 7, 10, 1],
		[0, 0, 4, 1],
		[0, 0, 0, 1],
	]
	clusters = [[0], [1], [2]]
	clusters, cli = h.merge (clusters, distance)
	assert cli == 2
	assert len (clusters) == 3
	assert len (clusters[0]) == 1 and clusters[0][0] == 0
	assert len (clusters[2]) == 0
	assert len (clusters[1]) == 2 and clusters[1][0] == 1 and clusters[1][1] == 2 

@pytest.mark.skip
def test_measure_linkage ():
	h = Hierarchical (KL2Mean, None)
	cl1 = [0,2]
	cl2 = [1]
	distance = [
		[0, 3, 1, 1],
		[0, 0, 4, 1],
		[0, 0, 0, 1],
	]
	linkage = h.measure_linkage (cl1, cl2, distance)
	assert linkage == (3+4)/2

@pytest.mark.skip
def test_KL2Mean_measure ():
	C = ['c1', 'c2']
	V = ['a', 'b', 'c', 'd']
	count = [[1,2,3,4],[5,6,7,8]]
	h = Hierarchical (KL2Mean, None)
	cp,jp,wp,cpp = h.estimate_p (count)
	distance = h.measure_distance (cp,jp,wp, V, C)
	assert False # not test yet. Just confirm it works by looking at the result.

