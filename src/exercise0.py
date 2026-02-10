import json
from anytree import Node, RenderTree, AsciiStyle
from anytree.importer import DictImporter
importer = DictImporter()
data = json_tree
root = importer.import_(data)
output = RenderTree(root)
print(output)
