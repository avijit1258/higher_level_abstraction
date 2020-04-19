import ast

class PlayingWithAST:
    """ This class takes a python file as input returns its function name with corresponding docstring"""

    def file_to_function_docstring_pair(self, filename):
        """ This function returns function name with their docstring """

        function_with_docstring = {}
        with open(filename) as f:
            tree = ast.parse( f.read() ,filename= filename, mode='exec')

            for node in ast.walk(tree):
                # print(type(node))
                if isinstance(node, ast.ClassDef):
                    for cn in ast.iter_child_nodes(node):
                        if isinstance(cn, ast.FunctionDef):
                            # print(ast.dump(cn))
                            function_with_docstring[cn.name] = ast.get_docstring(cn)
                            # print(cn.name)
                            # print(ast.get_docstring(cn))
                    # print('Class definition',ast.get_docstring(node))
        return function_with_docstring

# for node in ast.walk(parsed_tree):
# # print(ast.get_docstring(node))
#     print(ast.dump(node))

# for n in ast.iter_child_nodes(node):
#     if isinstance(n, ast.FunctionDef):
#         print(n.body)

# # if isinstance(node, ast.Module):
# #         print(node.body)
# for cn in ast.iter_child_nodes(node):
#     if isinstance(cn, ast.FunctionDef):
#         print(cn.name)


# pwa = PlayingWithAST()

# print(pwa.file_to_function_docstring_pair('/home/avb307/projects/higher_level_abstraction/tool/ClusteringCallGraph.py'))