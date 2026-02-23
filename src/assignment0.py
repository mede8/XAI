import json
from anytree import Node, RenderTree, AsciiStyle
from anytree.importer import JsonImporter


def main():
    importer = JsonImporter()
    with open('coffee.json', 'r') as file:
        data = file.read()
    root = importer.import_(data)
    output = RenderTree(root)
    print(output)


if __name__ == "__main__":
    main()
