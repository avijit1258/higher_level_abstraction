import ast
import glob
import os


class PlayingWithAST:
    """ This class takes a python file as input returns its function name with corresponding docstring"""

    function_with_docstring = {}

    def file_to_function_docstring_pair(self, filename):
        """ This function returns function name with their docstring """

        with open(filename) as f:
            tree = ast.parse(f.read(), filename=filename, mode='exec')

            for node in ast.walk(tree):
                # print(type(node))
                if isinstance(node, ast.ClassDef):
                    for cn in ast.iter_child_nodes(node):
                        if isinstance(cn, ast.FunctionDef):
                            # print(ast.dump(cn))
                            self.function_with_docstring[cn.name] = self.process_docstring(
                                ast.get_docstring(cn))
                            # print(cn.name)
                            # print(ast.get_docstring(cn))
                    # print('Class definition',ast.get_docstring(node))
                if isinstance(node, ast.FunctionDef):
                    self.function_with_docstring[node.name] = self.process_docstring(
                        ast.get_docstring(node))
        return

    def process_docstring(self, doc):
        print('Comment in:', doc)
        if doc is None:
            return ''
        for line in doc.split('\n'):
            line = line.strip()
            if (line == "") or (not any([c.isalnum() for c in line])):
                continue
            print('Comment out:', line)
            if '.' in line:
                return line
            else:
                return line+'.'

        return ''

    def get_all_py_files(self, root):
        all_py = []
        for path, subdirs, files in os.walk(root):
            for name in files:
                if name.endswith('.py'):
                    all_py.append(os.path.join(path, name))
                    print(os.path.join(path, name))

        return all_py

    def get_all_method_docstring_pair_of_a_project(self, location):
        all_files = self.get_all_py_files(location)

        for f in all_files:
            self.file_to_function_docstring_pair(f)
        
        return self.function_with_docstring

    def get_function_to_comment_ratio(self, location):

        self.get_all_method_docstring_pair_of_a_project(location)
        total_methods = len(self.function_with_docstring)
        no_comments = 0
        for method, doc in self.function_with_docstring.items():
            if doc == '':
                no_comments += 1

        return (no_comments/total_methods) * 100


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
# print(pwa.file_to_function_docstring_pair('/home/avb307/projects/hla_dataset/Real-Time-Voice-Cloning/vocoder/audio.py'))

# print(pwa.get_all_method_docstring_pair_of_a_project('/home/avb307/projects/hla_dataset/Real-Time-Voice-Cloning'))
