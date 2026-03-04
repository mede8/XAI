from anytree import AnyNode, RenderTree
from anytree.importer import JsonImporter


def check_norm(node: AnyNode, norm: dict) -> bool:
    """Annotates the tree with a violation status.

    :param node: the current node to check.
    :param norm: the rulebook to check against.
    :return: True if the node violates the norm, False otherwise.
    """
    # check the leaves first
    if node.is_leaf:
        in_list = node.name in norm['actions']
        if norm['type'] == 'P':
            # prohibition: violation if the action is in the list
            node.violation = in_list

        elif norm['type'] == 'O':
            # obligation: violation if the action is not in the list
            node.violation = not in_list

        return node.violation

    # check the children recurseively
    child_results = []
    for child in node.children:
        child_results.append(check_norm(child, norm))

    # we determine now the current node status based on children and the node type (aggregate)
    node_type = getattr(node, 'type', 'SEQ')

    if norm['type'] == 'P':
        if node_type == 'OR':
            # violation only if all children violate the norm
            node.violation = all(child_results)
        else:
            # for SEQ/AND, if any child violates the norm, the node is a violation
            node.violation = any(child_results)

    elif norm['type'] == 'O':
        # check for satisfaction
        # Is true if any child is satisfied
        is_satisfied = any(not res for res in child_results)
        node.violation = not is_satisfied

    return node.violation


def main():
    importer = JsonImporter()

    norm = {"type": "P", "actions": ["gotoKitchen"]}
    norm = {"type": "O", "actions": ["gotoShop", "gotoKitchen"]}

    try:
        with open('coffee.json', 'r') as file:
            data = file.read()
        root = importer.import_(data)
        check_norm(root, norm)

        results = []
        for pre, fill, node in RenderTree(root):
            res = f"{pre}{node!r}"
            results.append(res)
        output = "\n".join(results)
        print(output)

    except FileNotFoundError:
        print("Error: 'coffee.json' file not found.")


if __name__ == "__main__":
    main()
