class CKY:
	@staticmethod
	def recognize (words, G):
		# built parse table
		table = CKY._gen_parse_table (words)
		wnum = len (words)
		for j in range (1, wnum+1):
			CKY._collect_pos (j, words, G, table)
			CKY._collect_constituents (j, words, G, table)
		return table

	@staticmethod	
	def parse (parse_table): pass
	 	# later. Not have any solution yet.		

	@staticmethod
	def evaluate (): pass

	@staticmethod
	def _collect_pos (j, words, G, table):
		# collect non-terminals of a constituent
		w = words[j-1]
		for k,v in G['lexicon'].items ():
			if w in v:
				if table[j-1][j] is None: table[j-1][j] = []
				table[j-1][j].append (CKY._gen_entry (k))

	@staticmethod			
	def _collect_constituents (j, words, G, table):
		# find split points, 
		# conbine the right-side non-terminals, 
		# and collect their corresponding left-side non-terminals
		for i in range (0, j-1): # 0 < i <= j-2
			i = j-2-i
			for k in range (i+1, j): # i < k <= j-1
				left = table[i][k]
				right = table[k][j]
				if left is None or right is None: continue
				for li in left:
					for ri in right:
						li_nt = list (li.keys ())[0]
						ri_nt = list (ri.keys ())[0]
						possible_nt ='{} {}'.format (li_nt, ri_nt)
						for l,nt in G['rules'].items ():
							if possible_nt in nt: 
								if table[i][j] is None: table[i][j] = []
								leftr, leftl, leftri = i, k, left.index (li)
								rightr, rightc, rightri = k, j, right.index (ri)
								table[i][j].append (CKY._gen_entry (l, leftr, leftl, leftri, rightr, rightc, rightri))

	@staticmethod
	def _gen_parse_table (words):
		wnum = len (words)
		parse_table = [[None for i in range (wnum + 1)] for j in range (wnum + 1)]
		return parse_table

	@staticmethod	
	def _gen_entry (l, leftr=None, leftl=None, leftri=None, rightr=None, rightc=None, rightri=None):
		# leftr, leftl: row and column index of left cell
		# rightr, rightc: row and column index of right cell
		# leftri, rightri: index of nt in the lists of nt of cell left and right respectively
		return {l: [[leftr, leftl, leftri], [rightr, rightc, rightri]]}




class CKY1 (CKY):
	# variant of CKY. Dealing with unit producition directly