from collections import defaultdict
import os, sys
sys.path.insert (0, os.getcwd ())
import cfg.cnf

CNF = cfg.cnf.CNF

class PCNF (CNF):
	# Not convert unit production to CNF. Let parser deal directly
	@staticmethod
	def to_CNF (cfg, p_table, up=False):
		dcount = 0 # dummy count
		G = cfg['rules']
		L = cfg['lexicon']
		cnf = {
			'rules': defaultdict (lambda: [], {}),
			'lexicon': L
		}

		for l,r in G.items ():
			temp = [ri for ri in r]
			while not PCNF._is_valid (cnf['rules'][l], L, up):
				for ri in temp:
					prev_dcount = dcount
					i = temp.index (ri)
					if up: PCNF._resolve_unit_production (ri, l, G, cnf) # should never run
					dcount = PCNF._resolve_mixed (dcount, ri, i, l, cnf['rules'], G, L)
					if dcount > 0 and dcount > prev_dcount:
						PCNF._update_mixed_dm_p (p_table, prev_dcount, dcount)
						prev_dcount = dcount		
					dcount = PCNF._resolve_more2_nonterminal (dcount, ri, i, l, G, cnf)
					if dcount > 0 and dcount > prev_dcount:
						PCNF._update_more2nt_dm_p (p_table, prev_dcount, dcount)
						prev_dcount = dcount										
				temp = [ri for ri in cnf['rules'][l]]
		return cnf, p_table

	@staticmethod	
	def _update_mixed_dm_p (p_table, prev_dcount, dcount):
		# update p in the new dummy rules created after resolving mixed rules or more-than-2-nt rules
		for i in range (prev_dcount + 1, dcount + 1):
			dname = 'X{}'.format (i)
			p_table['lexicon'][dname] = [1]

	@staticmethod	
	def _update_more2nt_dm_p (p_table, prev_dcount, dcount):
		# update p in the new dummy rules created after resolving mixed rules or more-than-2-nt rules
		for i in range (prev_dcount + 1, dcount + 1):
			dname = 'X{}'.format (i)
			p_table['rules'][dname] = [1]