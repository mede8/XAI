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
    action_names = [node.name for node in trace if node.is_leaf]
    if norm['type'] == 'P':
        return any(action in norm['actions'] for action in action_names)
    elif norm['type'] == 'O':
        return not any(action in norm['actions'] for action in action_names)
    return False


def is_trace_feasible(trace, initial_beliefs, goal):
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
    for idx in preference_order:
        if costs1[idx] < costs2[idx]:
            return True
        elif costs1[idx] > costs2[idx]:
            return False
    return False


def main():
    importer = JsonImporter()

    norm = {"type": "P", "actions": ["payShop"]}
    goal = ["haveCoffee"]
    beliefs = ["staffCardAvailable", "ownCard", "colleagueAvailable",
               "haveMoney", "AnnInOffice"]
    preferences = [["quality", "price", "time"], [1, 2, 0]]

    with open('coffee.json', 'r') as file:
        data = file.read()
    root = importer.import_(data)

    if isinstance(norm, list):
        norm = norm[0]

    print("norm type:", type(norm), "value:", norm)

    all_traces = computeTraces(root, [])

    valid_traces = []
    for trace in all_traces:
        if not is_norm_violated(trace, norm) and is_trace_feasible(trace, beliefs, goal):
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

    print(output)


if __name__ == "__main__":
    main()