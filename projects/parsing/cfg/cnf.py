from collections import defaultdict

class CNF:
	@staticmethod
	def to_CNF (cfg, up=True):
		dcount = 0 # dummy count
		G = cfg['rules']
		L = cfg['lexicon']
		cnf = {
			'rules': defaultdict (lambda: [], {}),
			'lexicon': L
		}

		for l,r in G.items ():
			temp = [ri for ri in r]
			while not CNF._is_valid (cnf['rules'][l], L, up):
				for ri in temp:
					i = temp.index (ri)
					if up: CNF._resolve_unit_production (ri, l, G, cnf)
					dcount = CNF._resolve_mixed (dcount, ri, i, l, cnf['rules'], G, L)
					dcount = CNF._resolve_more2_nonterminal (dcount, ri, i, l, G, cnf)
				temp = [ri for ri in cnf['rules'][l]]
		return cnf

	@staticmethod	
	def _is_valid (r, L, up):
		valid = True
		if len (r) == 0: valid = False
		for nt in r:
			if CNF._is_mixed (nt, L) or CNF._has_more_2nt (nt) or (False if not up else CNF._has_up (nt)):
				valid = False
				break
		return valid

	@staticmethod
	def _is_mixed (ri, L):
		is_mixed = False
		ri_split = ri.split (' ')
		for k,v in L.items ():
			for t in v:
				for rij in ri_split:
					if t == rij:
						is_mixed = True
						return is_mixed
		return is_mixed

	@staticmethod	
	def _has_more_2nt (ri):
		hasmore2 = False
		if len (ri.split (' ')) > 2: hasmore2 = True
		return hasmore2
   
	@staticmethod
	def _has_up (ri):
		hasup = False
		if len (ri.split (' ')) == 1: hasup = True
		return hasup

	@staticmethod
	def _resolve_mixed (dcount, ri, i, l, cnfg, G, L):
		# FIX: one possible issue if parse using the result of such function is that, the same terminal will have many different dummy non-terminals refer to. This may reduce the relation between the terminal with sourounding words.
		if not CNF._is_mixed (ri, L): 
			if len (cnfg[l]) == 0: cnfg[l] = G[l] 
			return dcount
		ri_split = ri.split (' ')
		cnfg[l] = [i for i in G[l]]
		for rij in ri_split:
			if CNF._is_mixed (rij, L):
				dname = CNF._get_existing_dummy (rij, L)
				if dname is None: dcount, dname = CNF._gen_dummy_nt (dcount)
				L[dname] = [rij]
				rij_index = ri_split.index (rij)
				ri_split[rij_index] = dname
				ri_index = G[l].index (ri) 
				cnfg[l][ri_index] = ' '.join (ri_split)
		return dcount

	@staticmethod
	def _resolve_more2_nonterminal (dcount, ri, i, l, G, cnf):
		if not CNF._has_more_2nt (ri): 
			if len (cnf['rules'][l]) == 0: cnf['rules'][l] = G[l] 
			return dcount
		cnf['rules'][l] = [i for i in G[l]] if G.get (l, None) is not None and len (cnf['rules'][l]) == 0 else cnf['rules'][l]
		ri_split = ri.split (' ')
		dpart = ri_split[:-1]
		rest = ri_split[-1]
		sub_ri = ' '.join (dpart)
		dname = CNF._get_existing_dummy (sub_ri, cnf['rules'])
		if dname is None: dcount, dname = CNF._gen_dummy_nt (dcount)	
		cnf['rules'][dname] = [sub_ri]
		i = cnf['rules'][l].index (ri)
		cnf['rules'][l][i] = '{} {}'.format (dname, rest)
		if len (dpart) > 2:
			dcount = CNF._resolve_more2_nonterminal (dcount, sub_ri, 0, dname, G, cnf)
		return dcount

	@staticmethod
	def _gen_dummy_nt (dcount):
		name = 'X'
		dcount += 1
		dname = '{}{}'.format (name, dcount)
		return dcount, dname

	@staticmethod
	def _get_existing_dummy (ri, G):
		for l,r in G.items ():
			if len (r) == 1 and ri in r and 'X' == l[0]:
				return l
		return None 

	@staticmethod
	def _resolve_unit_production (ri, l, G, cnf):
		# assume CFG is complete. No non-terminal, if not being mapped to a word in the lexicon, has no rule
		# replace unit production with non-unit production. 
		# one the way to iterate the chain of unit productions, fix any one found.
		if not CNF._has_up (ri): 
			if len (cnf['rules'][l]) == 0: cnf['rules'][l] = G[l]
			return
		cnf['rules'][ri] = [x for x in G.get (ri, [])]
		for nt in G.get (ri, []):
			newl = ri
			newri = nt
			CNF._resolve_unit_production (newri, newl, G, cnf)

		if CNF._is_terminal (ri, cnf['lexicon']):
			cur_r_t = cnf['lexicon'][ri]
			cur_l_t = cnf['lexicon'].get (l, [])
			cur_l_t = list (set (cur_r_t + cur_l_t))
			cnf['lexicon'][l] = cur_l_t

		cnf['rules'][l] = [x for x in G[l]] if len (cnf['rules'][l]) == 0 else cnf['rules'][l]
		newnt = cnf['rules'][l] + cnf['rules'][ri]
		newnt = list (set (newnt))
		if ri in cnf['rules'][l]:
			del cnf['rules'][l][cnf['rules'][l].index (ri)]
		cnf['rules'][l] = [x for x in newnt if x is not ri]

	@staticmethod
	def _is_terminal (t, L):
		# check if the right side is a terminal
		ist = False
		if L.get (t, None) is not None: ist = True
		return ist

	@staticmethod
	def _has_unit_production (l, G):
		has_up = False
		for	ri in G[l]:
			if len (ri.split (' ')) == 1: 
				has_up = True
				break
		return has_up



