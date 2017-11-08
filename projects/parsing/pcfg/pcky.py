class PCKY:
	@staticmethod
	def recognize (words, G, p_table):
		# built parse table
		table = PCKY._gen_parse_table (words)
		wnum = len (words)
		for j in range (1, wnum+1):
			PCKY._collect_pos (j, words, G, table, p_table)
			PCKY._collect_constituents (j, words, G, table, p_table)
		return table		

	@staticmethod
	def parse (): pass

	@staticmethod
	def _collect_pos (j, words, G, table, p_table):
		# collect non-terminals of a constituent
		w = words[j-1]
		for k,v in G['lexicon'].items ():
			if w in v:
				if table[j-1][j] is None: table[j-1][j] = []
				w_index = v.index (w)
				p = p_table['lexicon'][k][w_index]
				table[j-1][j].append (PCKY._gen_entry (k, p=p))

	@staticmethod			
	def _collect_constituents (j, words, G, table, p_table): pass


	@staticmethod
	def _gen_parse_table (words):
		wnum = len (words)
		parse_table = [[None for i in range (wnum + 1)] for j in range (wnum + 1)]
		return parse_table

	@staticmethod	
	def _gen_entry (l, leftr=None, leftl=None, leftri=None, rightr=None, rightc=None, rightri=None, p=None): 
		# leftr, leftl: row and column index of left cell
		# rightr, rightc: row and column index of right cell
		# leftri, rightri: index of nt in the lists of nt of cell left and right respectively
		return {l: [[leftr, leftl, leftri], [rightr, rightc, rightri]], 'p': p}

	@staticmethod	
	def _estimate_p (): pass


