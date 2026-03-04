from anytree import AnyNode, RenderTree, search
from anytree.importer import DictImporter
from anytree.importer import JsonImporter


def computeTraces(current_node: AnyNode, current_trace: list[str]) -> list[list[str]]:
    """Recursively builds all possible execution traces from a given node.

    For OR nodes, each child becomes its own separate branch. For SEQ/AND
    nodes, all children are combined sequentially, which can still produce
    multiple branche if any child is itself an OR node further down the tree.

    :param current_node: the node from which to start the trace.
    :param current_trace: the trace
    :return: a list of all possible execution traces.
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


def is_norm_violated(trace: list[AnyNode], norm: dict) -> bool:
    """Checks whether a given trace violates a norm.

    For prohibitions (P), the trace violates if any action in it appears in the
    norm's action list. For obligations (O), the trace violates if none of the
    required actions appear anywhere in the trace.
    """
    action_names = [node.name for node in trace if node.is_leaf]
    if norm['type'] == 'P':
        return any(action in norm['actions'] for action in action_names)
    elif norm['type'] == 'O':
        return not any(action in norm['actions'] for action in action_names)
    return False


def is_trace_feasible(trace: list[AnyNode], initial_beliefs: list[str], goal: list[str]) -> bool:
    """Simulates executing a trace step by step and checks if it's actually
    doable.

    At each step, it verifies that the node's preconditions are satisfied given
    the current beliefs, then updates beliefs with the node's postconditions.
    Returns True only if the whole trace runs without any precondition failures
    and the goal is satisfied at the end.
    """
    beliefs = set(initial_beliefs)
    for node in trace:
        for condition in getattr(node, 'pre', []):
            if condition not in beliefs:
                return False
        beliefs.update(getattr(node, 'post', []))
    return all(g in beliefs for g in goal)


def compute_trace_cost(trace: list[AnyNode]) -> list[float]:
    """Sums up the cost vectors of all action nodes in a trace.

    Each action node has a list of costs, one per value dimension (e.g.
    quality, price, time). This function adds them all up and returns a
    single combined cost vector for the whole trace.
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


def is_preferred(costs1: list[float], costs2: list[float], preference_order: list[int]) -> bool:
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


""" - Pairilearn
importer = DictImporter()
root = importer.import_(json_tree)

if isinstance(norm, list):
    norm = norm[0]

all_traces = computeTraces(root, [])

valid_traces = [
    trace for trace in all_traces
    if not is_norm_violated(trace, norm)
    and is_trace_feasible(trace, beliefs, goal)
]

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
"""

if __name__ == "__main__":

    importer = JsonImporter()
    with open('coffee.json', 'r') as file:
        data = file.read()
    root = importer.import_(data)

    norm = {"type": "P", "actions": ["payShop"]}
    goal = ["haveCoffee"]
    beliefs = ["staffCardAvailable", "ownCard", "colleagueAvailable",
               "haveMoney", "AnnInOffice"]
    preferences = [["quality", "price", "time"], [1, 2, 0]]

    all_traces = computeTraces(root, [])
    valid_traces = [
        trace for trace in all_traces
        if not is_norm_violated(trace, norm)
        and is_trace_feasible(trace, beliefs, goal)
    ]

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
