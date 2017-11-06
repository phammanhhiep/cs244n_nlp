from collections import defaultdict

def to_CNF (CFG):
	dcount = 0 # dummy count
	G = CFG['rules']
	L = CFG['lexicon']
	CNF = {
		'rules': defaultdict (lambda: [], {}),
		'lexicon': L
	}

	for l,r in G.items ():
		temp = [ri for ri in r]
		while not _is_valid (CNF['rules'][l], L):
			for ri in temp:
				i = temp.index (ri)
				_resolve_unit_production (ri, l, G, CNF)
				dcount = _resolve_mixed (dcount, ri, i, l, CNF['rules'], G, L)
				dcount = _resolve_more2_nonterminal (dcount, ri, i, l, G, CNF)
			temp = [ri for ri in CNF['rules'][l]]
	return CNF

def _is_valid (r, L):
	valid = True
	if len (r) == 0: valid = False
	for nt in r:
		if _is_mixed (nt, L) or _has_more_2nt (nt) or _has_up (nt):
			valid = False
			break
	return valid

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

def _has_more_2nt (ri):
	hasmore2 = False
	if len (ri.split (' ')) > 2: hasmore2 = True
	return hasmore2

def _has_up (ri):
	hasup = False
	if len (ri.split (' ')) == 1: hasup = True
	return hasup

def _resolve_mixed (dcount, ri, i, l, CNF_G, G, L):
	# FIX: one possible issue if parse using the result of such function is that, the same terminal will have many different dummy non-terminals refer to. This may reduce the relation between the terminal with sourounding words.
	if not _is_mixed (ri, L): 
		if len (CNF_G[l]) == 0: CNF_G[l] = G[l] 
		return dcount
	ri_split = ri.split (' ')
	CNF_G[l] = [i for i in G[l]]
	for rij in ri_split:
		if _is_mixed (rij, L):
			dcount, dname = _gen_dummy_nonterminal (dcount)
			L[dname] = [rij]
			rij_index = ri_split.index (rij)
			ri_split[rij_index] = dname
			ri_index = G[l].index (ri) 
			CNF_G[l][ri_index] = ' '.join (ri_split)
	return dcount

def _resolve_more2_nonterminal (dcount, ri, i, l, G, CNF):
	if not _has_more_2nt (ri): 
		if len (CNF['rules'][l]) == 0: CNF['rules'][l] = G[l] 
		return dcount
	CNF['rules'][l] = [i for i in G[l]] if G.get (l, None) is not None and len (CNF['rules'][l]) == 0 else CNF['rules'][l]
	dcount, dname = _gen_dummy_nonterminal (dcount)
	ri_split = ri.split (' ')
	dpart = ri_split[:-1]
	rest = ri_split[-1]
	sub_ri = ' '.join (dpart)
	CNF['rules'][dname] = [sub_ri]
	i = CNF['rules'][l].index (ri)
	CNF['rules'][l][i] = '{} {}'.format (dname, rest)
	if len (dpart) > 2:
		dcount = _resolve_more2_nonterminal (dcount, sub_ri, 0, dname, G, CNF)
	return dcount

def _gen_dummy_nonterminal (dcount):
	dcount += 1
	dname = 'Dummy_{}'.format (dcount)
	return dcount, dname

def _resolve_unit_production (ri, l, G, CNF):
	# assume CFG is complete. No non-terminal, if not being mapped to a word in the lexicon, has no rule
	# replace unit production with non-unit production. 
	# one the way to iterate the chain of unit productions, fix any one found.
	if not _has_up (ri): 
		if len (CNF['rules'][l]) == 0: CNF['rules'][l] = G[l]
		return
	CNF['rules'][ri] = [x for x in G.get (ri, [])]
	for nt in G.get (ri, []):
		newl = ri
		newri = nt
		_resolve_unit_production (newri, newl, G, CNF)

	if _is_terminal (ri, CNF['lexicon']):
		cur_r_t = CNF['lexicon'][ri]
		cur_l_t = CNF['lexicon'].get (l, [])
		cur_l_t = list (set (cur_r_t + cur_l_t))
		CNF['lexicon'][l] = cur_l_t

	CNF['rules'][l] = [x for x in G[l]] if len (CNF['rules'][l]) == 0 else CNF['rules'][l]
	newnt = CNF['rules'][l] + CNF['rules'][ri]
	newnt = list (set (newnt))
	if ri in CNF['rules'][l]:
		del CNF['rules'][l][CNF['rules'][l].index (ri)]
	CNF['rules'][l] = [x for x in newnt if x is not ri]

def _is_terminal (t, L):
	# check if the right side is a terminal
	ist = False
	if L.get (t, None) is not None: ist = True
	return ist

def _has_unit_production (l, G):
	has_up = False
	for	ri in G[l]:
		if len (ri.split (' ')) == 1: 
			has_up = True
			break
	return has_up



