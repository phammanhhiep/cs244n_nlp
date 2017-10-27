# NEEDFIX: need to implement. Right now only the interface
class BeamSearch:
	def __init__ (self): pass
	
	def decoding (self, source_sent, phrase_tran_table, stack_size=None, stack_alpha=None, best_n=None):
		translation_options = self.get_translation_options (source_sent, phrase_tran_table)
		future_cost_table = self.get_future_cost_table (translation_options)
		stacks = []
		stack_num = len (source_sent)
		for s in range (stack_num):
			word_num = s + 1
			stacks[s] = []
			prev_stack = self._get_prev_stack (word_num, stacks) 
			new_hypotheses = self.create_hypotheses (word_num, translation_options)
			expanded_hypotheses = self.expanded_hypotheses (prev_stack)
			stacks[s].extend (new_hypotheses)
			stacks[s].extend (expanded_hypotheses)
			stacks[s] = self.get_translation_cost (stacks[s], future_cost_table)
			stacks[s] = self.prune_hypo (stacks[s], stack_size, stack_alpha)
		best_translations = self.get_n_best_translation (stacks, best_n)
		return best_translations
				
	def _get_prev_stack (self, word_num, stacks): pass	

	def get_translation_options (self, source_sent, phrase_tran_table): pass		

	def create_hypotheses (self, word_num, translation_options): pass

	def expand_hypotheses (self): pass

	def get_future_cost (self): pass

	def prune_hypo (self, stack, stack_size, stack_alpha):
		stack = self.recombine_hypo (stack)
		stack = self.threshold_prune (stack, stack_alpha)
		stack = self.histogram_prune (stack, stack_size)
		return stack

	def recombine_hypo (self, stack): pass

	def threshold_prune (self, stack, stack_alpha): pass

	def histogram_prune (self, stack, stack_size): pass

	def get_future_cost_table (self, translation_options): pass

	def get_translation_cost (self, stack, future_cost_table): pass
			
	def get_n_best_translation (stacks, best_n): pass	