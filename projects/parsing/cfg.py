def to_CNF (CFG):
	dcount = 0 # dummy count
	G = CFG['rules']
	L = CFG['lexicon']
	CNF = {
		'rules': {},
		'lexicon': L
	}

	for l,r in G.items ():
		for ri in r:
			i = r.index (ri)
			if ri is None: continue
			if len (ri.split (' ')) == 1: 
				_resolve_unit_production (ri, l, G, CNF)
			elif len (ri.split (' ')) > 2:
				if _is_mixed (ri, L):
					dcount = _resolve_mixed (dcount, ri, i, l, G, L)
				dcount = _resolve_more2_nonterminal (dcount, ri, i, l, G)
		_clean_none (r, l, G)

def _clean_none (r, l, G):
	# remove None elements in the right side of a rule
	# remove the rule if has not right side after remove None
	newr = []
	for ri in r:
		if ri is not None: newr.append (ri)
	G[l] = newr

def _is_mixed (ri, L):
	is_mixed = False
	for k,v in L.items ():
		for t in v:
			if t in ri: 
				is_mixed = True
				return is_mixed
	return is_mixed

def _resolve_mixed (dcount, ri, i, l, G, L):
	# FIX: one possible issue if parse using the result of such function is that, the same terminal will have many different dummy non-terminals refer to. This may reduce the relation between the terminal with sourounding words.
	ri_split = ri.split (' ')
	for rij in ri_split:
		if _is_mixed (rij, L):
			dcount, dname = _gen_dummy_nonterminal (dcount)
			L[dname] = [rij]
			rij_index = ri_split.index (rij)
			ri_split[rij_index] = dname
			G[l][i] = ' '.join (ri_split)
	return dcount

def _resolve_more2_nonterminal (dcount, ri, i, l, G):
	dcount, dname = _gen_dummy_nonterminal (dcount)
	ri_split = ri.split (' ')
	dpart = ri_split[:-1]
	rest = ri_split[-1]
	sub_ri = ' '.join (dpart)
	G[dname] = [sub_ri]
	i = G[l].index (ri)
	G[l][i] = '{} {}'.format (dname, rest)
	if len (dpart) > 2:
		dcount = _resolve_more2_nonterminal (dcount, sub_ri, 0, dname, G)
	return dcount

def _gen_dummy_nonterminal (dcount):
	dcount += 1
	dname = 'Dummy_{}'.format (dcount)
	return dcount, dname

def _resolve_unit_production (ri, l, G, CNF):
	# assume CFG is complete. No non-terminal, if not being mapped to a word in the lexicon, has no rule
	# replace unit production with non-unit production. 
	# one the way to iterate the chain of unit productions, fix any one found.
	cur_r = ri
	cur_l = l
	while True:
		if _is_terminal (cur_r, CNF['lexicon']):
			cur_r_t = CNF['lexicon'][cur_r]
			cur_l_t = CNF['lexicon'].get (cur_l, [])
			cur_l_t = list (set (cur_r_t + cur_l_t))
			CNF['lexicon'][cur_l] = cur_l_t
			i = G[cur_l].index (cur_r)
			G[cur_l][i] = None
			break
		else:
			if _has_unit_production (cur_r, G):
				for nt in G[cur_r]:
					if nt is not None and len (nt.split (' ')) == 1:
						_resolve_unit_production (nt, cur_r, G, CNF)
 
			i = G[cur_l].index (cur_r)
			G[cur_l][i] = None
			cur_l_nt = G[cur_l]
			for nt in G[cur_r]:
				if nt not in cur_l_nt and nt is not None:
					G[cur_l].append (nt)
			break

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



