from collections import defaultdict
from graphviz import *
import DFA

class NFA: 

	def __init__(self, states, alphabet, transitions, start, accept):
		self.states = states
		self.alphabet = alphabet
		# map from (state, transition letter) -> [states]
		self.transitions = transitions
		# array of starting states
		self.start = start
		self.accept = accept
		self.inverse_trans = defaultdict(list)
		for k, v in self.transitions.items():
			for s in v:
				self.inverse_trans[(s, k[1])].append(k[0])

	def next_state(self, letter, current):

		if (current, letter) in self.transitions:
			# list 
			return self.transitions[(current, letter)]

		return None


	# determine if accept or reject
	# do not change any values of object
	# if true returns a list of paths that are valid
	def simulate(self, string):
		paths = []
		for s in self.start:
			self.find_paths(string, [s], 0, paths)
		return paths

	# DFS for valid paths
	def find_paths(self, string, cur_path, cur_str_index, paths):

		if cur_str_index == len(string) and cur_path[-1] in self.accept:
		    paths.append(cur_path)
		    return
		if cur_str_index == len(string):
		    return

		next_states = self.next_state(string[cur_str_index], cur_path[-1])
		
		if next_states:
			for s in next_states:
				self.find_paths(string, cur_path + [s], cur_str_index + 1, paths)
		



	def convert_to_dfa(self):
		### convert here
		dfa_table = {}
		

		## state of states that have been seen
		seen = self.start

		added_to_table = set()

		while len(seen) > 0:

			cur_state = seen.pop(0)
			added_to_table.add(cur_state)

			for l in self.alphabet:
				state_set = set()
				## just a single state not a subset
				if (len(cur_state) == 1) and (cur_state[0], l) in self.transitions:
					for dest in self.transitions[(cur_state[0], l)]:
						state_set.add(dest)
				else:
					for s in cur_state:
						if (s, l) in self.transitions:
							for dest in self.transitions[(s, l)]:
								state_set.add(dest)
				
				dfa_table[(cur_state, l)] = state_set

				if tuple(state_set) not in added_to_table and tuple(state_set) not in seen:
						seen.append(tuple(state_set))

		dfa_states = set()
		dfa_accept = []
		dfa_transitions = {}
		for k, v in dfa_table.items():

			source_str = ",".join(k[0])
			dest_str = ",".join(v)

			## add appropriate accepting states
			for accept_state in self.accept:
				if accept_state in k[0]:
					if source_str not in dfa_accept:
						dfa_accept.append(source_str)
				if accept_state in v:
					if dest_str not in dfa_accept:
						dfa_accept.append(dest_str)

			dfa_states.add(source_str)
			dfa_states.add(dest_str)

			dfa_transitions[(source_str, k[1])] = dest_str


		dfa = DFA(dfa_states, self.alphabet, dfa_transitions, self.start, dfa_accept)
		return dfa



	# draws the current node green
	# if stopped draws current node red
	# if completed draws current node green
	def get_viz(self, current = None, stopped = False, completed = False):

		g = Digraph()

		if stopped:
			color = 'red'
		else:
			if completed:
				color = 'green'
			else:
				color = 'yellow'


		if current is None or current in self.start:
			start_color = color
			for s in self.start:
				if s in self.accept:
					g.node(s, shape='doublecircle', color=start_color, style='filled')
				else:
					g.node(s, color=start_color, style='filled')

		else:
			for s in self.start:
				if s in self.accept:
					g.node(s, shape='doublecircle')
				else:
					g.node(s)


		for s in self.accept:
			if s not in self.start:
				if s == current:
					g.node(s, color = color, shape='doublecircle', style='filled')
				else:
					g.node(s, shape='doublecircle')

		for s in self.states:
			if s not in self.accept and s not in self.start:
				if s == current:
					g.node(s, s, color=color, style='filled')
				else:
					g.node(s, s)

		edges = defaultdict(list)
		for k, v in self.transitions.items():
			## because nfa 
			for s in v:
				edges[(k[0], s)].append(k[1])

		for k, v in edges.items():
			label = "  " + ", ".join(v)
			g.edge(k[0],k[1], label)
		
		return g


	def get_svg(self, current = None, stopped = False, completed = False):

		viz = self.get_viz(current, stopped, completed)
		g = viz.unflatten()
		return g.pipe(format='svg').decode('utf-8')

