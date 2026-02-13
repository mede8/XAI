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
    action_names = [node.name for node in trace if node.is_leaf]
    if norm['type'] == 'P':
        return any(action in norm['actions'] for action in action_names)
    elif norm['type'] == 'O':
        return not any(action in norm['actions'] for action in action_names)
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

    print(output)


if __name__ == "__main__":
    main()
