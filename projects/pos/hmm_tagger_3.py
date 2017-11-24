import os, sys
sys.path.insert (0, os.getcwd ())
import hmm_tagger

HMM_Tagger = hmm_tagger.HMM_Tagger
Preprocessing = hmm_tagger.Preprocessing

class HMM_Tagger_3 (HMM_Tagger_2): pass
	# Trigram tagger
	# Smoothing using deleted interpolation
	# Handle unknown words 
	# Adding more features.