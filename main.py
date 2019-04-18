import networkx as nx
import matplotlib.pyplot as plt
import pydot
#import networkx.drawing.nx_pydot.read_dot

#caller_callee = [('F1', 'F3'), ('F1', 'F7'), ('F3', 'F7'), ('F4', 'F1')]
#self calling edge is omitted for now

# caller callee for generating
caller_callee = [('F0', 'F1'), ('F0', 'F2'), ('F0', 'F3'), ('F1', 'F3'), ('F2', 'F0'), ('F2', 'F1'), ('F4', 'F2'), ('F5', 'F1')]

S = []

T = []

execution_paths = []
G = nx.DiGraph()
#G.add_edges_from(caller_callee)
#G = nx.drawing.nx_agraph.read_dot('/pyan/pyan/pyan.dot')
#G = nx.DiGraph(nx.drawing.nx_pydot.read_dot('pyan/pyan/pyan.dot'))
#G = nx.read_graphml('pyan/pyan/pyan_yed.txt')
#nx.draw(G, with_labels=True)
#plt.show()


def extracting_source_and_exit_node():

    for s, v in G.in_degree():
        if v == 0:
            S.append(s)
            print(s)
    for t, v in G.out_degree():
        if v == 0:
            T.append(t)
            print(t)
def tgf_to_networkX():

        f = open("pyan/pyan/pyan_tgf.txt", "r")
        graph_started = False
        for line in f:
            if line.find('#') != -1:
                print(line.find('#'))
                graph_started = True
                continue

            if graph_started == True:
                edge_info = line.split()
                print(edge_info)
                G.add_edge(edge_info[0], edge_info[1])

        nx.draw(G, with_labels=True)
        plt.show()


        return

def extracting_execution_paths():

    for s in S:
        for t in T:
            #print(list(nx.dfs_preorder_nodes(G, s)))
            #print(list(nx.all_simple_paths(G, s, t)))
            #execution_paths.append(list(nx.all_simple_paths(G, s, t)))
            unpack_path = list(nx.all_simple_paths(G, s, t))
            for p in unpack_path:
                execution_paths.append(p)




tgf_to_networkX()
extracting_source_and_exit_node()
extracting_execution_paths()

print(execution_paths)
