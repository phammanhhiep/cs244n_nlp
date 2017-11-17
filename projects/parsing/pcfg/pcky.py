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
		# FIX: deal with unit production 
		# collect non-terminals of a constituent
		w = words[j-1]
		for k,v in G['lexicon'].items ():
			if w in v:
				if table[j-1][j] is None: table[j-1][j] = []
				wi = v.index (w)
				rule = PCKY._gen_pos_pointer (k, wi=wi)
				PCKY._estimate_p (p_table, rule)
				table[j-1][j].append (rule)

	@staticmethod			
	def _collect_constituents (j, words, G, table, p_table):
		# FIX: deal with unit production
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
						valid_rules = PCKY._collect_rules (li, ri, G['rules'], p_table)
						for rule in valid_rules:
							if table[i][j] is None:
								table[i][j] = [rule]
							elif (table[i][j][0]['p'] < rule['p']):
								table[i][j][0] = rule

	@staticmethod
	def _gen_parse_table (words):
		wnum = len (words)
		parse_table = [[None for i in range (wnum + 1)] for j in range (wnum + 1)]
		return parse_table

	@staticmethod	
	def _gen_pos_pointer (nt, wi=None): 
		rule = PCKY._gen_back_pointer (nt, wi=wi)
		return rule

	def _gen_back_pointer (nt, left=None, right=None, p=None, wi=None):
		return {'head': nt, 'left': left, 'right': right, 'p': p, 'wi': wi}

	@staticmethod	
	def _estimate_p (p_table, rule):
		# QA: should set p is None or zero?
		p = None 
		head = rule['head']
		index = rule['wi']
		if rule['left'] is None and rule['right'] is None: # pos
			rule['p'] = p_table['lexicon'][head][index]
		elif rule['left'] is None and rule['right'] is not None: # unit production 
			rule['p'] = p_table['rules'][head][index] * rule['right']['p']
		elif rule['left'] is not None and rule['right'] is not None: # standard rule 
			rule['p'] = p_table['rules'][head][index] * rule['left']['p'] * rule['right']['p'] 

	def _collect_up (nt, G, p_table, pointer):
		# return all nt pointers of unit productions
		up = []
		for l,v in G.items ():
			if nt in v:
				nti = v.index (nt)
				l_pointer = PCKY._gen_back_pointer (l, right=pointer, wi=nti)
				PCKY._estimate_p (p_table, l_pointer)
				new_up = PCKY._collect_up (l, G, p_table, l_pointer)
				up.append (l_pointer)
				up.extend (new_up)
		return up
							
	def _collect_rules (left_pointer, right_pointer, G, p_table):
		# Return a list of pairs of pointers and its associations. Each pair is the right-side of a rule. 

		left_up = PCKY._collect_up (left_pointer['head'], G, p_table, left_pointer)
		righ_up = PCKY._collect_up (right_pointer['head'], G, p_table, right_pointer)
		left_up.insert (0, left_pointer)
		righ_up.insert (0, right_pointer)
		rules = []

		for li in left_up:
			for ri in righ_up:
				possible_nt = '{} {}'.format (li['head'], ri['head'])
				for l,v in G.items ():
					if possible_nt in v:
						i = v.index (possible_nt)
						rule = PCKY._gen_back_pointer (l, li, ri, wi=i)
						PCKY._estimate_p (p_table, rule)
						rules.append (rule)

		return rules

	def traverse_parse_tree (): pass
		# Given a parse tree, retrive all component parses	




