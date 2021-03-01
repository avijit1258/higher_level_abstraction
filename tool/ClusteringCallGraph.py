import networkx as nx
import matplotlib.pyplot as plt
import pydot
import csv
import re
import pandas as pd
#import networkx.drawing.nx_pydot.read_dot
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, to_tree
from scipy.spatial.distance import pdist

from collections import defaultdict
from scipy.spatial import distance as ssd
import queue
import math
from timeit import default_timer as timer
import multiprocessing


from PlayingWithAST import *

from DocumentNodes import DocumentNodes
import config
import util

ROOT = config.ROOT
SUBJECT_SYSTEM_NAME = config.SUBJECT_SYSTEM_NAME
OUTPUT_DIRECTORY = ROOT + '/output/'
DATASET = ROOT + '/dataset/'+SUBJECT_SYSTEM_NAME+'.txt'
SUBJECT_SYSTEM_FOR_COMMENT = config.SUBJECT_SYSTEM_FOR_COMMENT # put location of repository for getting comments

document_nodes = DocumentNodes(OUTPUT_DIRECTORY, SUBJECT_SYSTEM_NAME)

class ClusteringCallGraph:
    """ This class takes caller-callee relationships of a python project. Next, builds a call graph from the input.
        Extracts execution paths from the call graph. Then clusters the execution paths according to their similarity.
        Finally, clusters are renamed using the execution paths under them using different topic modelling techniques.
    
     """
    S = []
    T = []
    tree = []
    text_data = []
    subject_system = ''
    special_functions = ['lambda', 'genexpr', 'listcomp', 'setcomp', 'dictcomp']
    execution_paths = []
    G = nx.DiGraph()
    function_id_to_name = {}
    function_id_to_file_name = {}

    pwa = PlayingWithAST()

    function_name_to_docstring = pwa.get_all_method_docstring_pair_of_a_project(SUBJECT_SYSTEM_FOR_COMMENT)



    def __del__(self):
        """ deletes the ClusteringCallGraph class objects """
        print('deleted')

    def python_analysis(self):
        """ analyzing python programs to build cluster tree of execution paths. """

        self.tgf_to_networkX()
        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        self.extracting_source_and_exit_node()
        start = timer()
        self.extracting_execution_paths()
        end = timer()
        print('Time required for extracting_execution_paths: ', end - start)
        print('No. of execution paths', len(self.execution_paths))

        
        if len(self.execution_paths) > 5000:
            self.execution_paths = util.random_sample_execution_paths(self.execution_paths)

        # self.remove_redundant_ep()
        
        start = timer()
        mat = self.distance_matrix(self.execution_paths)
        end = timer()
        print('Time required for distance_matrix: ', end - start)
        
        self.G.clear()

        document_nodes.initalize_graph_related_data_structures(self.execution_paths, self.function_id_to_name, \
            self.function_id_to_file_name, self.id_to_sentence, self.function_name_to_docstring)

        return self.clustering_using_scipy(mat)
        
    
    def check_ep_overlap_from_start(self, e, f):
        '''This function checks whether 2nd list is a sublist starting from start of 1st list'''
    
        for i in range(len(f)):
            if e[i] != f[i]:
                return False
                
        return True

    def remove_redundant_ep(self):
        ''' this function removes redundant execution paths from list of execution paths.
            for example, execution_paths = [['A', 'B', 'C', 'D'], ['B', 'C', 'D'], ['E', 'F', 'G'], ['I', 'F', 'S'], ['A', 'B'], ['A','B', 'C']]
            this list as input will produce a list [['A', 'B', 'C', 'D'], ['B', 'C', 'D'], ['E', 'F', 'G'], ['I', 'F', 'S']]
        
         '''

        self.execution_paths.sort(key = len, reverse = True)

        redundant_ep = []
        for e in self.execution_paths:
            if e in redundant_ep:
                continue
            for f in self.execution_paths:
                if e != f:
                    # print(e, f)
                    if self.check_ep_overlap_from_start(e, f):
                        redundant_ep.append(f)
        for r in redundant_ep:
            self.execution_paths.remove(r)
            
        print(' Filtered execution path length ', len(self.execution_paths))


    def tgf_to_networkX(self):
        """ converting tgf file to a networkX graph"""
        self.subject_system = SUBJECT_SYSTEM_NAME + '.txt' 
        
        f = open(DATASET, "r")
        G = nx.DiGraph()
        graph_started = False
        for line in f:

            if line.find('#') != -1:
                graph_started = True
                continue

            if graph_started == True:
                edge_info = line.split()
                # if self.function_id_to_name[edge_info[0]] in self.special_functions or self.function_id_to_name[edge_info[1]] in self.special_functions:
                #     continue
                # filter module calls from function calls
                if edge_info[0] in self.function_id_to_name and edge_info[1] in self.function_id_to_name:
                    self.G.add_edge(edge_info[0], edge_info[1])

            if graph_started == False and '.py' in line:
                ln = line.split(' ')
                self.function_id_to_name[ln[0]] = self.extract_function_name(ln[1])
                self.function_id_to_file_name[ln[0]] = line.split('/')[-1].split(':')[0]
        print('Function id to function name', self.function_id_to_name, 'len : ', len(self.function_id_to_name))        
        print('Function id to file name, function name', self.function_id_to_file_name)
        nx.draw(self.G, with_labels=True)
        plt.savefig(OUTPUT_DIRECTORY+'call-graph.png')
        plt.show()

        return

    def extracting_source_and_exit_node(self):
        """ Finding source and exit nodes from networkX graph """
        print('In degree')
        for s, v in self.G.in_degree():
            # print(s, v)
            if v == 0:
                self.S.append(s)
                # print(s)
        print(len(self.S))
        print('Out degree')
        for t, v in self.G.out_degree():
            # print(t, v)
            if v == 0:
                self.T.append(t)
                # print(t)

        print(len(self.T))

    def extracting_execution_paths(self):
        """ Extracting execution paths from networkX call graph """
        print('Extracting execution paths')
        for s in self.S:
            unpack_path = list(nx.all_simple_paths(self.G, s, self.T))
            for p in unpack_path:
                self.execution_paths.append(p)

        print("Number of EP: ", len(self.execution_paths))

    def distance_matrix(self, paths):
        """ creating distance matrix using jaccard similarity value """
        print('distance_matrix')
        length = len(paths)
        Matrix = [[0 for x in range(length)] for y in range(length)]
        for i in range(len(paths)):
            for j in range(len(paths)):
                # Matrix[i][j] = self.jaccard_similarity(paths[i], paths[j])
                Matrix[i][j] = util.compare_execution_paths(paths[i], paths[j])
                
        return Matrix



    def clustering_using_scipy(self, mt):
        """ clustering execution paths using scipy """

        start = timer()
        Z = linkage(ssd.squareform(mt), 'ward')
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z, truncate_mode='lastp', p=200)
        rootnode, nodelist = to_tree(Z, rd=True)
        nodes_with_parent = self.bfs_with_parent(nodelist, rootnode.id, math.ceil(math.log(len(nodelist) + 1, 2)))
        nodes_with_leaf_nodes = util.find_leaf_nodes_for_nodes(rootnode, nodelist)
        end = timer()
        print('Time required for clustering: ', end - start)
        
        count = 0
        
        start = timer()
        for child, parent in nodes_with_parent.items():
            if nodelist[child].count == 1:
                self.tree.append({'key': child, 'parent': parent, 'tfidf_word': 'EP: '+ str( child) \
                    + ', Name: ' +self.pretty_print_leaf_node(self.execution_paths[child]), \
                    'tfidf_method': '', 'lda_word': '', 'lda_method': '', 'lsi_word': '',\
                     'lsi_method': '', 'spm_method': '','text_summary': 'hello summary', 'files': [], 'files_count': 0})
                continue
            execution_paths_of_a_cluster = nodes_with_leaf_nodes[child]
            
            count += 1
            print('Cluster no: ', count)
            
            self.tree.append(document_nodes.labeling_cluster(execution_paths_of_a_cluster, child, parent))
            
        end = timer()
        print('Time required for labeling using 6 techniques', end - start)
        
        print(self.tree, file=open(OUTPUT_DIRECTORY+ 'TREE_DICT_' +self.subject_system, 'w'))

        return self.tree

    def extract_function_name(self,str):
        """ extracting function names from TGF file """
        end = str.find('\\')

        return str[:end]

    def similarity(self, list1, list2):
        print('list1 :', list1)
        print('list2 :', list2)
        print('braycurtis ', ssd.braycurtis(list1, list2))
        return ssd.braycurtis(list1, list2)


    def bfs_leaf_node(self, nodelist, id):
        """ 
        Finding leaf nodes of a node in cluster tree.
        """

        # node = nodelist[id]

        count = 0
        visited = [0] * len(nodelist)
        q = queue.Queue()
        q.put(id)
        visited[id] = 1
        leaf_nodes = []
        while True:
            if q.empty():
                break
            p = q.get()
            count = count + 1
            visited[p] = 1

            if nodelist[p].count == 1:
                leaf_nodes.append(p)
                continue

            if visited[nodelist[p].left.id] == 0:
                q.put(nodelist[p].left.id)
            if visited[nodelist[p].right.id] == 0:
                q.put(nodelist[p].right.id)

            # if count == nodelist[id].count:
            #     break

        return leaf_nodes

    
    def cluster_view(self, Z, dend):
        """ 
        Generate cluster figure from a dendogram.
        """
        X = self.flatten(dend['icoord'])
        Y = self.flatten(dend['dcoord'])
        leave_coords = [(x, y) for x, y in zip(X, Y) if y == 0]

        # in the dendogram data structure,
        # leave ids are listed in ascending order according to their x-coordinate
        order = np.argsort([x for x, y in leave_coords])
        id_to_coord = dict(zip(dend['leaves'], [leave_coords[idx] for idx in order]))  # <- main data structure

        # ----------------------------------------
        # get coordinates of other nodes

        # this should work but doesn't:

        # # traverse tree from leaves upwards and populate mapping ID -> (x,y);

        # map endpoint of each link to coordinates of parent node
        children_to_parent_coords = dict()
        for i, d in zip(dend['icoord'], dend['dcoord']):
            x = (i[1] + i[2]) / 2
            y = d[1]  # or d[2]
            parent_coord = (x, y)
            left_coord = (i[0], d[0])
            right_coord = (i[-1], d[-1])
            children_to_parent_coords[(left_coord, right_coord)] = parent_coord

        # traverse tree from leaves upwards and populate mapping ID -> (x,y)
        root_node, node_list = to_tree(Z, rd=True)
        ids_left = range(len(dend['leaves']), len(node_list))
        # As I truncated the dendogram, actual code from stackoverflow created infinite loop
        # Fixing the code maually solved the problem. I stoping the while loop solved the problem
        # ids_left = range(3375, 6746)
        # print(ids_left)

        count = 1
        while count == 1:
            count = count + 1
            for ii, node_id in enumerate(ids_left):
                if node_list[node_id].is_leaf():
                    continue
                node = node_list[node_id]
                if (node.left.id in id_to_coord) and (node.right.id in id_to_coord):
                    left_coord = id_to_coord[node.left.id]
                    right_coord = id_to_coord[node.right.id]
                    id_to_coord[node_id] = children_to_parent_coords[(left_coord, right_coord)]

            # ids_left = [node_id for node_id in range(len(node_list)) if not node_id in id_to_coord]
            # print(ids_left)

        # plot result on top of dendrogram
        ax = plt.gca()
        for node_id, (x, y) in id_to_coord.items():
            if not node_list[node_id].is_leaf():
                ax.plot(x, y, 'ro')
                ax.annotate(str(node_id), (x, y), xytext=(0, -8),
                            textcoords='offset points',
                            va='top', ha='center')

        dend['node_id_to_coord'] = id_to_coord
        plt.savefig('clustering.png')
        plt.show()

        return

    def flatten(self, l):
        """ 
        Flattens a nested list.
        """
        return [item for sublist in l for item in sublist]

    def bfs(self, nodelist, id, depth):
        """ 
        BFS on cluster tree to get nodes from top to bottom approach with a depth limit.
        """
        # node = nodelist[id]
        nodes = []
        count = 0
        visited = [0] * len(nodelist)
        q = queue.Queue()
        q.put(id)
        visited[id] = 1
        while True:
            # print(list(q.queue))
            if q.empty():
                break
            q.qsize()
            p = q.get()
            q.qsize()
            count = count + 1

            if nodelist[p].count == 1:
                nodes.append(p)
                # print(p, ' ', nodelist[p].count)
                visited[p] = 1
                continue

            if visited[nodelist[p].left.id] == 0:
                q.put(nodelist[p].left.id)
            if visited[nodelist[p].right.id] == 0:
                q.put(nodelist[p].right.id)

            nodes.append(p)
            # print(p, ' ', nodelist[p].count)
            visited[p] = 1

            if math.ceil(math.log(count + 1, 2)) == depth:
                break

        return nodes

    def bfs_with_parent(self, nodelist, id, depth):
        """ 
        BFS to get parent nodes from cluster tree with their child nodes. Key of the returned dict is parent node and values are their child nodes.
        """
        # node = nodelist[id]
        nodes = []
        count = 0
        visited = [0] * len(nodelist)
        q = queue.Queue()
        q.put(id)
        tree = dict()
        tree[id] = -1
        visited[id] = 1
        while True:
            if q.empty():
                break
            q.qsize()
            p = q.get()
            q.qsize()
            count = count + 1

            if nodelist[p].count == 1:
                nodes.append(p)
                visited[p] = 1
                continue

            if visited[nodelist[p].left.id] == 0:
                tree[nodelist[p].left.id] = nodelist[p].id
                q.put(nodelist[p].left.id)
            if visited[nodelist[p].right.id] == 0:
                tree[nodelist[p].right.id] = nodelist[p].id
                q.put(nodelist[p].right.id)

            nodes.append(p)
            
            visited[p] = 1

            # if math.ceil(math.log(count + 1, 2)) == depth:
            #     break
        
        return tree

        return

    def id_to_sentence(self,execution_paths):
        """
        This function takes a single execution path and maps its function id with names. Returns printable sentence of a execution path.
        """
        str = ''

        for l in execution_paths:
            str += self.function_id_to_name[l]
            str += ' '

        return str
    
    def pretty_print_leaf_node(self,execution_paths):
        """
        This function takes a single execution path and maps its function id with names. Returns printable sentence of a execution path.
        """
        str = ''

        for l in execution_paths:
            str += self.function_id_to_name[l]
            if l != execution_paths[len(execution_paths)-1]:
                str += ' &rarr; '

        return str



c = ClusteringCallGraph()

c.python_analysis()

document_nodes.workbook.close()

