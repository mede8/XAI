from anytree import AnyNode, RenderTree, search
from anytree.importer import JsonImporter


def computeTraces(current_node: AnyNode, current_trace: list[str]) -> list[list[str]]:
    """Computes all traces from the current node to the leaf nodes.

    :param current_node: the node from which to start the trace.
    :param current_trace: the trace.
    :return: a list of all possible execution traces.
    """
    new_path = current_trace + [current_node.name]
    if current_node.is_leaf is True:
        # New path may be a separate branch or a cluster of paths/branches
        return [new_path]
    node_type = getattr(current_node, 'type', 'N/A')
    if node_type == "OR":
        # We need to split into a different branch for every child
        branching_traces = []
        for child in current_node.children:
            branching_traces += computeTraces(child, new_path)
        return branching_traces
    else:
        # Here we need to append the remaining paths to all already existing branches.
        # This can also add even more branches
        # if it finds more OR types to split branches
        # Wrapping it up in order to be able to use for to go through branches
        current_branches = [new_path]
        for child in current_node.children:
            new_traces = []
            # We only need an appending path for the current branches
            # (which can branch further)
            appending_branches = computeTraces(child, [])
            # We then look at all our existing traces and add the new found traces
            # to all of them sequentially
            for existing_branch in current_branches:
                # The number of current subbranches is basically multiplied
                # by the number of appending branches
                for new_branch in appending_branches:
                    new_traces.append(existing_branch + new_branch)
            current_branches = new_traces
        return new_traces


def main():
    traces = []
    importer = JsonImporter()
    starting_node_name = "getCoffee"
    with open('coffee.json', 'r') as file:
        data = file.read()
    root = importer.import_(data)
    target_node = search.find(root, lambda node: node.name == starting_node_name)
    print(target_node)
    output = computeTraces(target_node, traces)
    print(output)
    # DotExporter(root, nodenamefunc=format_node_label).to_picture("goal_tree.png")


if __name__ == "__main__":
    main()
