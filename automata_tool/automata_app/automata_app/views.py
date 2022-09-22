from automata_app import app
from flask import render_template, jsonify, request, url_for, redirect
from automata_app.forms import *
from collections import defaultdict
from automata_app.DFA import *
from automata_app.NFA import *


@app.route("/")
@app.route("/home")
def home():
	return render_template('index.html')

@app.route("/dfa", methods=["GET", "POST"])
def createDFA():

	form = DFAForm()

	states = form.states.data
	alphabet = form.alphabet.data
	start_state = form.start_state.data
	accept_states = form.accept_states.data


	if accept_states is None:
		form.accept_states.choices = [("", "")]

	if request.method == "POST":

		dfa_states = states.split(",")

		# fill form back out
		form.start_state.choices = [(s, s) for s in dfa_states]
		form.accept_states.choices = [(s, s) for s in dfa_states]

		alphabet = alphabet.split(",")

		transitions = {}
		html_transitions = {}
		for s in dfa_states:
			for l in alphabet:
				rule_id = s + ";" + l
				if len(request.form[rule_id]) > 0:
					transitions[(s, l)] = request.form[rule_id]
					html_transitions[rule_id] = request.form[rule_id]



		dfa = DFA(dfa_states, alphabet, transitions, start_state, accept_states)
		

		graph = dfa.get_svg()
		minimized_graph = None

		if form.minimize.data:

			minimized_dfa = dfa.moore_reduction()
			minimized_graph = minimized_dfa.get_svg()


		elif form.step.data:


			inp = form.string_input.data

			remaining = form.remaining.data
			path = form.path.data.split(",")

			# the first step
			if path[0] == "":
				cur_letter = inp[0]
				cur_state = start_state
				remaining = inp[1:]
				# make sure the first thing isnt empty string
				path = []
			else:
				cur_letter = remaining[0]
				cur_state = path[-1]
				remaining = remaining[1:]


			next_state = dfa.next_state(cur_letter, current= cur_state)

			path.append(next_state)
			form.path.data = ",".join(path)
			form.remaining.data = remaining

			if next_state:
				# nothing left in string
				if len(remaining) == 0:
					if next_state in accept_states:
						form.result.data = "ACCEPT"
						graph = dfa.get_svg(current = next_state, completed=True)
					else:
						form.result.data = "REJECT"
						graph = dfa.get_svg(current = next_state, stopped=True)
				else:
					graph = dfa.get_svg(current = next_state)

			else:
				form.result.data = "REJECT"
				graph = dfa.get_svg(current = cur_state, stopped = True)


			

		elif form.simulate.data:

			result, path = dfa.simulate(form.string_input.data)

			form.remaining.data = ""
			form.path.data = ",".join(path)
			form.result.data = "ACCEPT" if result else "REJECT"


			if len(path) == 0:
				graph = dfa.get_svg(current = start_state, stopped = True)
			else:
				last_state = path[-1]
				if result:
					graph = dfa.get_svg(current = last_state, completed = True)
				else:
					graph = dfa.get_svg(current = last_state, stopped = True)


		elif form.reset.data:

			form.string_input.data = ""
			form.result.data = ""
			form.path.data = ""
			form.remaining.data = ""



		return render_template('dfa.html', form = form, graph=graph, transitions = html_transitions, minimized_graph=minimized_graph)

	return render_template('dfa.html', form = form, transitions = {})

@app.route("/nfa", methods=["GET", "POST"])
def nfa():
	
	form = NFAForm()

	states = form.states.data
	alphabet = form.alphabet.data
	start_state = form.start_state.data
	accept_states = form.accept_states.data


	if accept_states is None:
		form.accept_states.choices = [("", "")]

	if request.method == "POST":

		nfa_states = states.split(",")

		# fill form back out
		form.start_state.choices = [(s, s) for s in nfa_states]
		form.accept_states.choices = [(s, s) for s in nfa_states]

		alphabet = alphabet.split(",")

		transitions = {}
		html_transitions = {}
		for s in nfa_states:
			for l in alphabet:
				rule_id = s + ";" + l
				if len(request.form[rule_id]) > 0:
					transitions[(s, l)] = request.form[rule_id].split(",")
					html_transitions[rule_id] = request.form[rule_id]



		nfa = NFA(nfa_states, alphabet, transitions, [start_state], accept_states)
		

		graph = nfa.get_svg()

		converted_graph = None
		

		if form.convert.data:
			dfa = nfa.convert_to_dfa()
			converted_graph = dfa.get_svg()

		elif form.simulate.data:

			paths = nfa.simulate(form.string_input.data)
			path_str = "\n".join([",".join(p) for p in paths])
			
			if len(paths) > 0:
				form.paths.data = path_str
			else:
				form.paths.data = "NONE"

			form.result.data = "ACCEPT" if len(paths) > 0 else "REJECT"

		elif form.reset.data:

			form.string_input.data = ""
			form.result.data = ""
			form.paths.data = ""


		return render_template('nfa.html', form = form, graph=graph, transitions = html_transitions, converted_graph=converted_graph)

	return render_template('nfa.html', form = form, transitions = {})

