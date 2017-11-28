import os,sys
sys.path.insert (0, os.getcwd ())
import pytest, math
import topic_classifier

TC = topic_classifier.Topic_Classifier

# pytest.mark.skip ()
def test_train (): pass

pytest.mark.skip ()
def test_count_params ():
	docs = [
		[['this', 'is', 'a', 'house'], ['c1', 'c2']],
		[['the', 'house', 'is', 'mine'], ['c2', 'c3']],
	]

	params = TC.count_params (docs)
	assert params['_stat_']['V']['count'] == 6
	assert params['_stat_']['doc']['count'] == 2
	assert params['C']['c1']['d_count'] == 1
	assert params['C']['c2']['d_count'] == 2
	assert params['C']['c3']['d_count'] == 1
	assert params['C']['c1']['f_count'] == 4
	assert params['C']['c2']['f_count'] == 8
	assert params['C']['c1']['features']['is']['count'] == 1
	assert params['C']['c2']['features']['is']['count'] == 2

	assert params['C']['c1']['notc_d_count'] == 2
	assert params['C']['c1']['notc_f_count'] == 8
	assert params['C']['c2']['features']['is']['notc_count'] == 2

pytest.mark.skip ()
def test_estimate_params ():
	docs = [
		[['this', 'is', 'a', 'house'], ['c1', 'c2']],
		[['the', 'house', 'is', 'mine'], ['c2', 'c3']],
	]
	
	params = TC.count_params (docs)
	params = TC.estimate_params (params)
	assert params['C']['c1']['log'] == math.log (1/2)
	assert params['C']['c2']['features']['is']['log'] == math.log ((2+1)/(8+6))
	assert params['C']['c1']['notc_log'] == math.log (1)
	assert params['C']['c2']['features']['is']['notc_log'] == math.log ((2+1)/(8+6))	

pytest.mark.skip ()
def test_classify ():
	docs = [[['this', 'ship', 'that', 'sailed'], [None]]]
	params = {	
		'C':{
			'c1': {
				'log': 2, 'notc_log': 3,
				'features': {
					'this': {'log': 1, 'notc_log': 2},
					'ship': {'log': 2, 'notc_log': 4},
					'that': {'log': 2, 'notc_log': 1},
					'sailed': {'log': 2, 'notc_log': 2},
				}
			},
			'c2': {
				'log': 1, 'notc_log':2,
				'features': {
					'this': {'log': 1, 'notc_log': 2},
					'ship': {'log': 4, 'notc_log': 1},
					'that': {'log': 2, 'notc_log': 2},
					'sailed': {'log': 1, 'notc_log': 1},
				}		
			}
		}
	}

	labels = TC.classify (docs, params)
	expected_labels = [['c2']]
	assert len (labels) == 1
	for i in range (len (labels[0])): assert labels[0][i] == expected_labels[0][i]

	params['C']['c2']['notc_log'] = 200 # to make sure no topic founded
	labels = TC.classify (docs, params)
	expected_labels = [[]]
	assert len (labels) == 1
	for i in range (len (labels[0])): assert labels[0][i] == expected_labels[0][i]	


pytest.mark.skip ()
def test_estimate_log_c ():
	ws = ['this', 'ship', 'that', 'sailed']
	cparam = {
		'log': 2, 'notc_log': 3,
		'features': {
			'this': {'log': 1, 'notc_log': 2},
			'ship': {'log': 2, 'notc_log': 4},
			'that': {'log': 2, 'notc_log': 1},
			'sailed': {'log': 2, 'notc_log': 2},
		}
	}

	c_log, notc_log = TC.estimate_log_c (ws, cparam)
	assert c_log == 9
	assert notc_log == 12

def test_evaluate (): pass



