from collections import defaultdict
from graphviz import *
import random
import datetime

class DFA: 

    def __init__(self, states, alphabet, transitions, start, accept):
        self.states = states
        self.alphabet = alphabet
        # map from (state, transition letter) -> state
        self.transitions = transitions
        self.start = start
        self.accept = accept
        self.current = start
        self.inverse_trans = defaultdict(list)
        for k, v in self.transitions.items():
            self.inverse_trans[(v, k[1])].append(k[0])

    def next_state(self, letter, current = None):

        if current is None:
            current = self.current

        if (current, letter) in self.transitions:
            return self.transitions[(current, letter)]

        return None


    # determine if accept or reject
    # do not change any values of object
    def simulate(self, string, start = None):

        if start is None:
            start = self.start

        next_state = None
        path = [start]
        curr = start
        for s in string:
            next_state = self.next_state(s, curr)
            if next_state:
                path.append(next_state)
                curr = next_state
            else:
                return (False, path)

        if curr in self.accept:
            return (True, path)
        return (False, path)


    def reverse(self):
    
        return NFA(self.states, self.alphabet, self.inverse_trans, self.accept, [self.start])
        


    def brzozowski(self):
        return (((self.reverse()).convert_to_dfa()).reverse()).convert_to_dfa()


    def build_minimized(self, P):
        new_state_names = []
        new_accept = []
        new_start = None

        for new_states in P:
            if len(new_states) == 0:
                continue
            accept = False
            start = False
            new_state = ""
            for s in new_states:
                if s in self.accept:
                    accept = True
                if s == self.start:
                    start = True
                new_state += str(s) + ","
            new_state = new_state[:-1]

            if accept:
                new_accept.append(new_state)
            if start:
                new_start = new_state

            new_state_names.append(new_state)

        new_transitions = {}
        for s in new_state_names:
            orig_state = s[0]
            for l in self.alphabet:
                if (orig_state, l) in self.transitions and self.transitions[(orig_state, l)]:
                    for poss_dest in new_state_names:
                        if self.transitions[(orig_state, l)] in poss_dest.split(","):
                            new_transitions[(s, l)] = poss_dest

        minimized_DFA = DFA(new_state_names, self.alphabet, new_transitions, new_start, new_accept)
        return minimized_DFA

    

    def split(self, C, I):
        return frozenset(C.intersection(I)), frozenset(C.difference(I))


    def hopcroft(self):
        P = set([frozenset(set(self.states) - set(self.accept)), frozenset(set(self.accept))])
        if len(set(self.states) - set(self.accept)) < len(set(self.accept)):
            M = set(self.states) - set(self.accept)
        else:
            M = set(self.accept)

        W = set()
        for letter in self.alphabet:
            W.add((frozenset(M), letter))

        
        new_W = set()
        while len(W) > 0:
            new_P = set()
            e = W.pop()
            S, l = e[0], e[1]

            # CREATE INVERSE SET
            I = set()
            for s in S:
                if (s, l) in self.inverse_trans:
                    for x in self.inverse_trans[(s, l)]:
                        I.add(x)
            
            # SEE WHAT SETS IN P ARE SPLIT
            for C in P:
                if len(C) == 0:
                    continue
                B1, B2 = self.split(C, I)
                if len(B1) > 0 and len(B2) > 0:
                    new_P.add(B1)
                    new_P.add(B2)
                    if len(B1) <= len(B2):
                        small_split = B1
                    else:
                        small_split = B2

                    for letter in self.alphabet:
                        if (C, letter) in W:
                            W.remove((C, letter))
                            new_W.add((B1, letter))
                            new_W.add((B2, letter))
                        W.add((small_split, letter))
                else:
                    new_P.add(C)

            P = new_P
    
        return self.build_minimized(P)

    

    def moore_reduction(self):
        P = [set(self.states) - set(self.accept), set(self.accept)]
        new_P = None
        n = 0
        while new_P != P:
            new_P = P
            for curr_set in new_P:
                for letter in self.alphabet:

                    I = set()
                    for s in curr_set:
                        if (s, letter) in self.inverse_trans:
                            for x in self.inverse_trans[(s, letter)]:
                                I.add(x)
                    # split the classes based on I
                    temp_P = []
                    for C in P:
                        common = C.intersection(I)
                        diff = C.difference(I)
                        if len(common) > 0 and len(diff) > 0:
                            temp_P.append(common)
                            temp_P.append(diff)
                        else:
                            temp_P.append(C)
                    P = temp_P

        return self.build_minimized(P)



    def trim(self):


        ## reachable 
        reachable_states = set()
        self.reachable_dfs(self.start, reachable_states)

        final_reach = set()
        ## see if reachable states can reach an accepting state
        for s in reachable_states:
            s_reach = set()
            self.reachable_dfs(s, s_reach)
            for d in s_reach:
                if d in self.accept:
                    final_reach.add(s)

        new_states = []
        new_transitions = {}

        for s in self.states:
            if s in final_reach:
                new_states.append(s)
        for k, v in self.transitions.items():
            if k[0] in final_reach and v in final_reach:
                new_transitions[k] = v

        new_accept = [a for a in self.accept if a in new_states]
        return DFA(new_states, self.alphabet, new_transitions, self.start, new_accept)


    def reachable_dfs(self, cur_state, visited):

        visited.add(cur_state)

        neighbors = set()
        for l in self.alphabet:
            if (cur_state, l) in self.transitions:
                if self.transitions[(cur_state, l)]:
                    neighbors.add(self.transitions[(cur_state, l)])

        for n in neighbors:
            if n not in visited:
               self.reachable_dfs(n, visited)


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


        if current is None or current == self.start:
            start_color = color
            if self.start in self.accept:
                g.node(self.start, shape='doublecircle', color=start_color, style='filled')
            else:
                g.node(self.start, color=start_color, style='filled')

        else:
            if self.start in self.accept:
                g.node(self.start, shape='doublecircle')
            else:
                g.node(self.start)


        for s in self.accept:
            if s != self.start:
                if s == current:
                    g.node(s, color = color, shape='doublecircle', style='filled')
                else:
                    g.node(s, shape='doublecircle')

        for s in self.states:
            if s not in self.accept and s != self.start:
                if s == current:
                    g.node(s, s, color=color, style='filled')
                else:
                    g.node(s, s)

        edges = defaultdict(list)
        for k, v in self.transitions.items():
            edges[(k[0], v)].append(k[1])

        for k, v in edges.items():
            label = "  " + ", ".join(v)
            g.edge(k[0],k[1], label)
        
        return g


    def get_svg(self, current = None, stopped = False, completed = False):
        viz = self.get_viz(current, stopped, completed)
        g = viz.unflatten()
        return g.pipe(format='svg').decode('utf-8')


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

        poss_next = []

        if (current, letter) in self.transitions:
            poss_next += self.transitions[(current, letter)]

        if len(poss_next) > 0:
            return list(set(poss_next))
        return None


    # determine if accept or reject
    # do not change any values of object
    # if true returns a list of paths that are valid
    def simulate(self, string):
        paths = []
        self.find_paths(string, [self.start], 0, paths)
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
        seen = set()

        orig_state_ind = {}
        i = 0
        for s in self.states:
            orig_state_ind[s] = i
            i += 1

        start_id = 0
        for s in self.start:
            start_id |= 1 << (orig_state_ind[s])

        seen.add(start_id)
        states = set()

        while len(seen) > 0:

            cur_state_id = seen.pop()
            states.add(cur_state_id)
            cur_states = [self.states[i] for i in range(len(self.states)) if (1 << i) & cur_state_id]

            for l in self.alphabet:
                
                dest_id = 0
                ## just a single state not a subset
                for s in cur_states:
                    if (s, l) in self.transitions:
                        for dest in self.transitions[(s, l)]:
                            dest_id |= 1 << (orig_state_ind[dest])


                if dest_id != 0:
                    dfa_table[(cur_state_id, l)] = dest_id
                    if dest_id not in states and dest_id not in seen:
                            seen.add(dest_id)
        dfa_states = states

        dfa_accept = set()
        for s in states:
            for q in self.accept:
                if q in self.states and (1 << orig_state_ind[q]) & s:
                    dfa_accept.add(s)

        dfa = DFA(list(dfa_states), self.alphabet, dfa_table, start_id, dfa_accept)
        return dfa


    # def convert_to_dfa_old(self):
    #     ### convert here
    #     dfa_table = {}
        
    #     if isinstance(self.start, list) or isinstance(self.start, tuple):
    #         start_str = ",".join(self.start)
    #     else:
    #         start_str = self.start
    #     ## state of states that have been seen
    #     seen = [start_str]

    #     added_to_table = set()

    #     while len(seen) > 0:

    #         cur_state_str = seen.pop(0)
    #         cur_state = cur_state_str.split(",")

    #         added_to_table.add(cur_state_str)
    
    #         for l in self.alphabet:
    #             state_set = set()
    #             ## just a single state not a subset
    #             if len(cur_state) == 1:
    #                 if (cur_state[0], l) in self.transitions:
    #                     for dest in self.transitions[(cur_state[0], l)]:
    #                         state_set.add(dest)
    #             else:
    #                 for s in cur_state:
    #                     if (s, l) in self.transitions:
    #                         for dest in self.transitions[(s, l)]:
    #                             state_set.add(dest)

    #             if len(state_set) == 0:
    #                 continue

    #             dest_state = ",".join(sorted(list(state_set)))
    #             dfa_table[(cur_state_str, l)] = dest_state

    #             if dest_state not in added_to_table and dest_state not in seen:
    #                     seen.append(dest_state)
    
    #     dfa_states = set()
    #     dfa_accept = []
    #     dfa_transitions = {}

    #     ### REMOVE DUPLICATE ROWS
    #     seen_rows = set()
    #     rows = {}
    #     deleted = set()
    #     new_states = set()
    #     converted = {}

    #     for s in added_to_table:
    #         all_letters = []
    #         for l in self.alphabet:
    #             if (s, l) in dfa_table:
    #                 all_letters.append(dfa_table[(s, l)])
    #             else:
    #                 all_letters.append("")
    #         if tuple(all_letters) in seen_rows:
    #             converted[s] = rows[tuple(all_letters)]
    #             deleted.add(s)
    #         else:
    #             seen_rows.add(tuple(all_letters))
    #             rows[tuple(all_letters)] = s
    #             converted[s] = s
    #             new_states.add(s)

    #     ## ensure relabling
    #     relab = {}
    #     i = 1
    #     for s in new_states:
    #         relab[s] = str(i)
    #         i += 1

    #     for k, v in dfa_table.items():          

    #         source_str = relab[converted[k[0]]]
    #         dest_str = relab[converted[v]]


    #         # add appropriate accepting states
    #         for accept_state in self.accept:
    #             if accept_state in k[0].split(","):
    #                 if source_str not in dfa_accept:
    #                     dfa_accept.append(source_str)
    #             if accept_state in v.split(","):
    #                 if dest_str not in dfa_accept:
    #                     dfa_accept.append(dest_str)


    #         if k[0] in deleted:
    #             continue

    #         dfa_states.add(source_str)
    #         dfa_states.add(dest_str)

    #         dfa_transitions[(source_str, k[1])] = dest_str

    #     dfa = DFA(list(dfa_states), self.alphabet, dfa_transitions, self.start, dfa_accept)
    #     return dfa



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
            if self.start in self.accept:
                g.node(s, shape='doublecircle')
            else:
                g.node(s)


        for s in self.accept:
            if s != self.start:
                if s == current:
                    g.node(s, color = color, shape='doublecircle', style='filled')
                else:
                    g.node(s, shape='doublecircle')

        for s in self.states:
            if s not in self.accept and s != self.start:
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

#EXAMPLE USAGE

states = ["0", "1", "2", "3", "4"]
alphabet = ["a", "b"]
transitions = {("0", "a"): "0", ("0", "b"): "1", 
              ("1", "a"): "2", ("1", "b"): "3",
              ("2", "a"): "2", ("2", "b"): "4", 
              ("3", "a"): "2", ("3", "b"): "3", 
              ("4", "a"): "4", ("4", "b"): "4"}
start = "0"
accept = ["0", "2", "4"]
dfa = DFA(states, alphabet, transitions, start, accept)

# EDGE CASES (BRZ)
# states = ["0", "1", "2"]
# alphabet = ["a", "b"]
# transitions = {("0", "a"): "1", ("0", "b"): "2", 
#               ("1", "a"): "1", ("1", "b"): "2",
#               ("2", "a"): "2", ("2", "b"): "2"}
# start = "0"
# accept = ["1"]
# dfa = DFA(states, alphabet, transitions, start, accept)


# states = ["0", "1"]
# alphabet = ["A", "B"]
# transitions = {("0", "A"): "1",
#                 ("1", "A"): "1"}
# start = "1"
# accept = ["1"]
# dfa = DFA(states, alphabet, transitions, start, accept)


# states = ["0", "1", "2"]
# alphabet = ["a", "b"]
# transitions = {("0", "a"): "1", ("0", "b"): "2", 
#               ("2", "a"): "0", ("2", "b"): "0"}
# start = "0"
# accept = ["2"]
# dfa = DFA(states, alphabet, transitions, start, accept)


# states = ["0", "1", "2"]
# alphabet = ["a", "b"]
# transitions = {("0", "b"): "0", ("1", "a"): "0", 
#               ("2", "a"): "1"}
# start = "0"
# accept = ["0"]
# dfa = DFA(states, alphabet, transitions, start, accept)


# states = ["0", "1", "2"]
# alphabet = ["a", "b"]
# transitions = {("0", "b"): "0", ("1", "a"): "0", 
#               ("2", "a"): "2", ("2", "b"): "0"}
# start = "1"
# accept = ["2"]
# dfa = DFA(states, alphabet, transitions, start, accept)

# states = ["0", "1", "2"]
# alphabet = ["a", "b"]
# transitions = {("0", "a"): "1", ("1", "a"): "2", 
#               ("2", "a"): "1", ("2", "b"): "2"}
# start = "1"
# accept = ["1"]
# dfa = DFA(states, alphabet, transitions, start, accept)




mini_1 = dfa.moore_reduction().trim()
print("MOR")
print(len(mini_1.transitions))
print(len(mini_1.states))

mini_2 = dfa.hopcroft().trim()
print("HOP")
print(len(mini_2.transitions))
print(len(mini_2.states))

mini_3 = dfa.brzozowski().trim()

print("BRZ")
print(len(mini_3.transitions))
print(len(mini_3.states))

print(mini_1.transitions)
print(mini_2.transitions)
print(mini_3.transitions)



def generateDFA(state_size, alphabet_size, t_prob):
    states = [str(i) for i in range(state_size + 1)]
    alphabet = [chr(65 + i) for i in range(alphabet_size)]
    transitions = {}
    for i in range(state_size):
        for j in range(alphabet_size):
            r = random.random()
            if r < t_prob:
                dest_ind = random.randint(0, state_size - 1)
                transitions[(states[i], alphabet[j])] = states[dest_ind]

    start_ind = random.randint(0, state_size - 1)
    start = states[start_ind]
    accept_ind = random.randint(0, state_size - 1)
    accept = [states[accept_ind]]



    return DFA(states, alphabet, transitions, start, accept)


def compare_mini(state_size, alphabet_size, t_prob, N):

    moore_times = []
    hop_times = []
    brz_times = []

    difference = []

    for i in range(N):

        dfa = generateDFA(state_size, alphabet_size, t_prob)

        t_1 = datetime.datetime.now()
        moore = dfa.moore_reduction()
        t_2 = datetime.datetime.now()


        moore_times.append((t_2 - t_1).total_seconds())

        t_1 = datetime.datetime.now()
        hop = dfa.hopcroft()
        t_2 = datetime.datetime.now()

        hop_times.append((t_2 - t_1).total_seconds())

        t_1 = datetime.datetime.now()
        brz = dfa.brzozowski()
        t_2 = datetime.datetime.now()

        brz_times.append((t_2 - t_1).total_seconds())

        moore = moore.trim()
        hop = hop.trim()
        brz = brz.trim()


        # ensure that the algorithms return same minimization
        if (len(brz.transitions) != len(hop.transitions) or len(brz.states) != len(hop.states)
            or len(brz.transitions) != len(moore.transitions) or len(brz.states) != len(moore.states)):
            difference.append((moore, hop, brz, dfa))


    return moore_times, hop_times, brz_times, difference 

moore_times, hop_times, brz_times, differences = compare_mini(20, 2, .8, 100)

moore_av = sum(moore_times) / len(moore_times)
hop_av = sum(hop_times) / len(hop_times)
brz_av = sum(brz_times) / len(brz_times)

print("MOORE AVE: " + str(moore_av))
print("HOP AVE: " + str(hop_av))
print("BRZ AVE: " + str(brz_av))




            






