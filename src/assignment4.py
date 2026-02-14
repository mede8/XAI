import json
from anytree import Node, RenderTree, search, SymlinkNode
from anytree.importer import JsonImporter

import json
from anytree import Node, RenderTree, search
from anytree.importer import JsonImporter


def computeTraces(current_node, current_trace):
    """Returns all possible execution traces as lists of node objects."""
    new_path = current_trace + [current_node]
    if current_node.is_leaf:
        return [new_path]
    node_type = getattr(current_node, 'type', 'N/A')
    if node_type == "OR":
        # split into a different branch for every child
        branching_traces = []
        for child in current_node.children:
            branching_traces += computeTraces(child, new_path)
        return branching_traces
    else:
        # SEQ / AND: append all children sequentially (can further branch)
        current_branches = [new_path]
        for child in current_node.children:
            new_traces = []
            appending_branches = computeTraces(child, [])
            for existing_branch in current_branches:
                for new_branch in appending_branches:
                    new_traces.append(existing_branch + new_branch)
            current_branches = new_traces

            
        return current_branches


def is_norm_violated(trace, norm):
    """Check if a trace violates a given norm.

    For a Prohibition (P): it's violated if any action in the trace is in
    norm['actions'].
    For an Obligation (O): it's violated if none of the actions in the trace
    are in norm['actions'].
    """
    names = [node.name.lower() for node in trace]
    # normalize norm actions once
    norm_actions = [a.lower() for a in norm['actions']]
    if norm['type'] == 'P':
        return any(a in norm_actions for a in names)
    elif norm['type'] == 'O':
        return not any(a in norm_actions for a in names)
    return False


def is_trace_feasible(trace, initial_beliefs, goal):
    """Simulate execution of the trace and return True if it is feasible and
    satisfies the goal.

    Feasibility requires that all preconditions of every node encountered
    in the trace are satisfied at the moment of execution (beliefs updated
    along the way).
    Goal satisfaction requires that all goal conditions hold after
    the trace completes.
    """
    beliefs = set(initial_beliefs)

    for node in trace:
        # check preconditions of every node in the trace
        pre = getattr(node, 'pre', [])
        for condition in pre:
            if condition not in beliefs:
                return False

        # update beliefs with postconditions
        post = getattr(node, 'post', [])
        for condition in post:
            beliefs.add(condition)

    return all(g in beliefs for g in goal)


def compute_trace_cost(trace):
    """Return the summed cost vector for all action nodes in the trace."""
    total_costs = None
    for node in trace:
        costs = getattr(node, 'costs', None)
        if costs is not None:
            costs = list(costs)
            if total_costs is None:
                total_costs = costs[:]
            else:
                for i in range(len(costs)):
                    total_costs[i] += costs[i]
    return total_costs if total_costs is not None else [0.0]


def is_preferred(costs1, costs2, preference_order):
    """Return True if costs1 is preferred (lower cost) over costs2.
    """
    for idx in preference_order:
        if costs1[idx] < costs2[idx]:
            return True
        elif costs1[idx] > costs2[idx]:
            return False
    return False  # costs are equal, no strict preference



def explain_action(trace, action_name, root, initial_beliefs, norm, preferences):    
    # empty if action not in trace
    if action_name not in trace:
        return []

    # convert trace to nodes
    trace_nodes = [search.find(root, lambda n: n.name == name) for name in trace]
    # index of action
    idx = trace.index(action_name)

    explanations = []
    beliefs = set(initial_beliefs)
    order = preferences[1]
    # for norm string in explanations
    norm_str = f"{norm['type']}({','.join(norm['actions'])})"

    # helper: which OR child is chosen for THIS action (direct child)
    def chosen_child_for(or_node):
        # pick the child whose subtree contains the action we’re explaining
        for child in or_node.children:
            # if child is in trace, it must be on the path to the action node 
            if child in trace_nodes or action_name in {leaf.name for leaf in child.leaves}:            
                return child
        return None
    
    # iterate through trace up to and including the action to explain
    for i in range(idx + 1):
        node = trace_nodes[i]        

        # P factors for action nodes (only if they have preconditions, otherwise no P factor)
        if node is not None and node.is_leaf:
            pre = getattr(node, "pre", [])
            if pre:  # spec: no P if no preconditions
                satisfied = [p for p in pre if p in beliefs]
                explanations.append(["P", node.name, satisfied])

            # update beliefs after executing ACT
            beliefs.update(getattr(node, "post", []))
            continue  # ACT handled; OR nodes shouldn’t also update beliefs here

        # OR node explanations (C, V, N)        
        if node is not None and getattr(node, "type", None) == "OR":
            chosen = chosen_child_for(node)                      
             
            if chosen is None:
                continue  # OR not relevant to action_name           
            
            # C factor
            pre_c = getattr(chosen, "pre", [])
            if pre_c:
                sat_c = [p for p in pre_c if p in beliefs]
                explanations.append(["C", chosen.name, sat_c])
            else:
                explanations.append(["C", chosen.name, []])

            # compute chosen cost (use one representative trace)
            chosen_traces = computeTraces(chosen, [])
            chosen_trace = chosen_traces[0] if chosen_traces else []
            cost_chosen = compute_trace_cost(chosen_trace)

            # pick best alternative competitor for V (most preferred among unchosen)
            best_alt = None
            best_alt_cost = None

            # also track first norm-violating alternative for N (if any)
            violating_alt = None

            for alt in node.children:

                if alt == chosen:
                    continue
                    
                alt_traces = computeTraces(alt, [])
                
                # explanations.append(["length alt traces =", len(alt_traces[0])])
                                
                if not alt_traces:
                    continue

                # N check: any trace violates?
                for t_idx, t in enumerate(alt_traces):                    
                    if is_norm_violated(t, norm) != False and violating_alt is None:
                            violating_alt = alt
                            break


                # V candidate: best cost among unchosen
                alt_cost = compute_trace_cost(alt_traces[0])
                if best_alt_cost is None or is_preferred(alt_cost, best_alt_cost, order):
                    best_alt = alt
                    best_alt_cost = alt_cost

            # V factor (rubric format uses ">" and compares chosen vs NOT chosen)
            if best_alt is not None:
                explanations.append(["V", chosen.name, cost_chosen, ">", best_alt.name, best_alt_cost])

            # N factor (only if some alternative violates)
            if violating_alt is not None:
                explanations.append(["N", violating_alt.name, norm_str])

            # F factor (failed condition)
            pre_alt = getattr(chosen, "pre", [])
            for condition in pre_alt:
                if condition is not None and condition not in beliefs:
                    not_sat_alt = [p for p in pre_alt if p not in beliefs]
                    explanations.append(["F", chosen.name, not_sat_alt])
            
        # L factor (linked node)
        if node is not None and isinstance(node, SymlinkNode):
            explanations.append(["L", node.name, node.target.name])
        
    
    # D factor (goal and subgoals from walking up the trace)
    action_node = trace_nodes[idx]
    seen=set()
    if action_node is not None:
        p = action_node.parent
        while p is not None:
            if getattr(p, "type", None) in {"OR", "SEQ", "AND"} or p is root:
                if p.name not in seen:
                    explanations.append(["D", p.name])
                    seen.add(p.name)
            p = p.parent

          
    # add U at the end
    explanations.append(["U", preferences[0], preferences[1]])

    return explanations


def main():
    importer = JsonImporter()

    norm = {"type": "P", "actions": ["PayShop"]}
    goal = ["haveCoffee"]
    beliefs = ["staffCardAvailable", "ownCard", "colleagueAvailable",
               "haveMoney", "AnnInOffice"]
    preferences = [["quality", "price", "time"], [1, 2, 0]]

    with open('coffee.json', 'r') as file:
        data = file.read()
    root = importer.import_(data)
    explanation = []

    # enumerate all possible traces (as node objects for access to
    # pre/post/costs)
    all_traces = computeTraces(root, [])

    # keep only traces that satisfy the norm and are feasible given the
    # beliefs and goal
    valid_traces = []
    for trace in all_traces:
        if not is_norm_violated(trace, norm) and is_trace_feasible(trace,
                                                                   beliefs,
                                                                   goal):
            valid_traces.append(trace)

    if not valid_traces:
        output = []
    else:
        # select the trace that minimises costs according to the user's
        # preference ordering
        # e.g.[1, 2, 0] → price > time > quality
        preference_order = preferences[1]

        best_trace = valid_traces[0]
        best_costs = compute_trace_cost(valid_traces[0])

        for trace in valid_traces[1:]:
            costs = compute_trace_cost(trace)
            if is_preferred(costs, best_costs, preference_order):
                best_trace = trace
                best_costs = costs

        output = [node.name for node in best_trace]
        action_to_explain = "getCoffeeKitchen"
        explanation = explain_action(output, action_to_explain, root, beliefs, norm, preferences)

    return explanation


if __name__ == "__main__":
    explanation = main()
    print(explanation)
   


