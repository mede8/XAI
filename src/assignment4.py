import json
from anytree import Node, RenderTree, search
from anytree.importer import DictImporter
import json
from anytree.importer import JsonImporter


def computeTraces(current_node, current_trace):
    """Recursively builds all possible execution traces from a given node.

    For OR nodes, each child becomes its own separate branch. For SEQ/AND
    nodes, all children are combined sequentially, which can still produce
    multiple branches if any child is itself an OR node further down the tree.
    """
    new_path = current_trace + [current_node]
    if current_node.is_leaf:
        return [new_path]
    node_type = getattr(current_node, 'type', 'N/A')
    if node_type == "OR":
        branching_traces = []
        for child in current_node.children:
            branching_traces += computeTraces(child, new_path)
        return branching_traces
    else:
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
    """Checks whether a given trace violates a norm.

    For prohibitions (P), the trace violates if any action in it appears in the
    norm's action list. For obligations (O), the trace violates if none of the
    required actions appear anywhere in the trace.
    """
    names = [node.name.lower() for node in trace]
    norm_actions = [a.lower() for a in norm['actions']]
    if norm['type'] == 'P':
        return any(a in norm_actions for a in names)
    elif norm['type'] == 'O':
        return not any(a in norm_actions for a in names)
    return False


def is_trace_feasible(trace, initial_beliefs, goal):
    """Simulates executing a trace step by step and checks if it's actually
    doable.

    At each step, it verifies that the node's preconditions are satisfied given
    the current beliefs, then updates beliefs with the node's postconditions.
    Returns True only if the whole trace runs without any precondition failures
    and the goal is satisfied at the end.
    """
    beliefs = set(initial_beliefs)
    for node in trace:
        pre = getattr(node, 'pre', [])
        for condition in pre:
            if condition not in beliefs:
                return False
        post = getattr(node, 'post', [])
        for condition in post:
            beliefs.add(condition)
    return all(g in beliefs for g in goal)


def compute_trace_cost(trace):
    """Sums up the cost vectors of all action nodes in a trace.

    Each action node has a list of costs, one per value dimension (e.g.
    quality, price, time). This function adds them all up and returns a single
    combined cost vector for the whole trace.
    """
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
    """Returns True if costs1 is strictly better than costs2 given the user's
    preferences.

    Compares costs lexicographically according to the preference order, where
    the first index in the list is the most important value. If all values are
    equal, returns False since neither is strictly better.
    """
    for idx in preference_order:
        if costs1[idx] < costs2[idx]:
            return True
        elif costs1[idx] > costs2[idx]:
            return False
    return False


def explain_action(trace, action_name, root, initial_beliefs, norm,
                   preferences):
    """Generates a structured explanation for why a specific action was
    performed.

    Walks through the selected trace and collects explanatory factors for each
    OR node (why one alternative was chosen over others), each action's
    preconditions, any links from the explained action, its ancestor goals,
    and the user's preferences. Returns an empty list if the action isn't in
    the trace.
    """
    if action_name not in trace:
        return []

    trace_nodes = [search.find(root,
                               lambda n,
                               name=name: n.name == name) for name in trace]
    idx = trace.index(action_name)

    explanations = []
    beliefs = set(initial_beliefs)
    order = preferences[1]
    norm_str = f"{norm['type']}({', '.join(norm['actions'])})"

    def chosen_child_for(or_node):
        for child in or_node.children:
            leaves_names = {leaf.name for leaf in child.leaves}
            if child in trace_nodes or action_name in leaves_names:
                return child
        return None

    def follow_link_chain(node, visited=None):
        if visited is None:
            visited = set()
        if node.name in visited:
            return
        visited.add(node.name)
        for linked_name in getattr(node, "link", []):
            explanations.append(["L", node.name, "->", linked_name])
            linked_node = search.find(root, lambda n,
                                      ln=linked_name: n.name == ln)
            if linked_node:
                follow_link_chain(linked_node, visited)

    for i in range(idx + 1):
        node = trace_nodes[i]

        if node is not None and node.is_leaf:
            pre = getattr(node, "pre", [])
            if pre:
                satisfied = [p for p in pre if p in beliefs]
                explanations.append(["P", node.name, satisfied])

            beliefs.update(getattr(node, "post", []))
            continue

        if node is not None and getattr(node, "type", None) == "OR":
            chosen = chosen_child_for(node)

            if chosen is None:
                continue

            pre_c = getattr(chosen, "pre", [])
            if pre_c:
                sat_c = [p for p in pre_c if p in beliefs]
                explanations.append(["C", chosen.name, sat_c])
            else:
                explanations.append(["C", chosen.name, []])

            chosen_subtrace = [n for n in trace_nodes if n is not None and
                               (n is chosen or chosen in n.ancestors)]
            cost_chosen = compute_trace_cost(chosen_subtrace)

            for alt in node.children:
                if alt == chosen:
                    continue

                alt_traces = computeTraces(alt, [])
                if not alt_traces:
                    continue

                norm_violated = any(is_norm_violated(t,
                                                     norm) for t in alt_traces)
                if norm['type'] == 'O' and (
                    alt.is_leaf or getattr(alt, 'type', '') == 'ACT'
                ):
                    norm_violated = False

                if norm_violated:
                    explanations.append(["N", alt.name, norm_str])
                    continue

                pre_alt = getattr(alt, "pre", [])
                not_sat_alt = [p for p in pre_alt if p not in beliefs]
                if not_sat_alt:
                    explanations.append(["F", alt.name, not_sat_alt])
                    continue

                alt_cost = compute_trace_cost(alt_traces[0])
                for t in alt_traces[1:]:
                    c = compute_trace_cost(t)
                    if is_preferred(c, alt_cost, order):
                        alt_cost = c

                explanations.append(["V", chosen.name, cost_chosen, ">",
                                     alt.name, alt_cost])

    action_node = trace_nodes[idx]
    follow_link_chain(action_node)

    seen = set()
    if action_node is not None:
        p = action_node.parent
        while p is not None:
            if getattr(p, "type", None) in {"OR", "SEQ", "AND"} or p is root:
                if p.name not in seen:
                    explanations.append(["D", p.name])
                    seen.add(p.name)
            p = p.parent

    explanations.append(["U", [preferences[0], preferences[1]]])

    return explanations


def main(json_tree, norm, goal, beliefs, preferences, action_to_explain):
    """Main entry point that selects the best trace and generates an
    explanation.

    Loads the goal tree, filters out traces that violate the norm or aren't
    feasible given the agent's beliefs, then picks the best remaining trace
    according to the user's preferences. Finally generates an explanation for
    the requested action within that trace.
    """
    importer = DictImporter()
    root = importer.import_(json_tree)
    explanation = []

    if isinstance(norm, list):
        norm = norm[0]
    norm_type = norm.get('type') or norm.get('norm_type') or norm.get('Type')
    norm_actions = norm.get('actions') or norm.get('norm_actions') or norm.get(
        'Actions') or []
    norm = {'type': norm_type, 'actions': norm_actions}

    all_traces = computeTraces(root, [])

    valid_traces = []
    for trace in all_traces:
        if not is_norm_violated(trace, norm) and is_trace_feasible(trace,
                                                                   beliefs,
                                                                   goal):
            valid_traces.append(trace)

    if not valid_traces:
        output = []
    else:
        preference_order = preferences[1]
        best_trace = valid_traces[0]
        best_costs = compute_trace_cost(valid_traces[0])

        for trace in valid_traces[1:]:
            costs = compute_trace_cost(trace)
            if is_preferred(costs, best_costs, preference_order):
                best_trace = trace
                best_costs = costs

        output = [node.name for node in best_trace]
        explanation = explain_action(output, action_to_explain, root, beliefs,
                                     norm, preferences)

    return output, explanation


if __name__ == "__main__":

    importer = JsonImporter()
    with open('coffee.json', 'r') as file:
        json_tree = json.load(file)

    norm = {"type": "P", "actions": ["payShop"]}
    goal = ["haveCoffee"]
    beliefs = ["staffCardAvailable", "ownCard", "colleagueAvailable",
               "haveMoney", "AnnInOffice"]
    preferences = [["quality", "price", "time"], [1, 2, 0]]
    action_to_explain = "getCoffeeKitchen"

    trace, explanation = main(json_tree, norm, goal, beliefs, preferences,
                              action_to_explain)
    print("Selected trace:", trace)
    print("Explanation:")
    for factor in explanation:
        print(factor)
