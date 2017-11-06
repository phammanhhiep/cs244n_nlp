def recognize (words, G):
	# built parse table
	table = _gen_parse_table (words)
	sindex = eindex = 0
	wnum = len (words)
	for j in range (1, wnum+1):
		_collect_pos (j, words, G, table)
		_collect_constituents (j, words, G, table)
	return table

def parse (recognizer): pass
	
def _collect_pos (j, words, G, table):
	# collect non-terminals of a constituent
	w = words[j-1]
	for k,v in G['lexicon'].items ():
		if w in v:
			if table[j-1][j] is None: table[j-1][j] = []
			table[j-1][j].append (k)

def _collect_constituents (j, words, G, table):
	# find split points, 
	# conbine the right-side non-terminals, 
	# and collect their corresponding left-side non-terminals
	for i in range (0, j-1): # 0 < i <= j-2
		i = j-2-i
		for k in range (i+1, j): # i < k <= j-1
			left = table[i][k]
			right = table[k][j]
			if left is None or right is None: break
			for li in left:
				for ri in right:
					possible ='{} {}'.format (li, ri)
					for l,nt in G['rules'].items ():
						if possible in nt: 
							if table[i][j] is None: table[i][j] = []
							if l not in table[i][j]: table[i][j].append (l)

def _gen_parse_table (words):
	wnum = len (words)
	parse_table = table = [[None for i in range (wnum + 1)] for j in range (wnum + 1)]
	return parse_table