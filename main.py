import networkx as nx
import matplotlib.pyplot as plt
import pydot
import csv
import re
import pandas as pd
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
    print('In degree')
    for s, v in G.in_degree():
        print(s, v)
        if v == 0:
            S.append(s)
            #print(s)
    print('Out degree')
    for t, v in G.out_degree():
        print(t, v)
        if v == 0:
            T.append(t)
            #print(t)


def extract_function_name(str):

    end = str.find('\\')


    return str[:end]


def tgf_to_networkX():

        f = open("pyan/pyan/pyan_tgf.txt", "r")
        function_id_to_name = {}
        graph_started = False
        for line in f:

            if line.find('#') != -1:
                #print(line.find('#'))
                graph_started = True
                continue

            if graph_started == True:
                edge_info = line.split()
                #print(edge_info)
                G.add_edge(function_id_to_name[edge_info[0]], function_id_to_name[edge_info[1]])

            if graph_started == False:
               ln = line.split(' ')
               #print(ln)
               function_id_to_name[ln[0]] = extract_function_name(ln[1])

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


def buildgraph(f, view):
    #g = nx.DiGraph()
    stack = []

    for line in f:
        # get func and file names without unnecessary texts
        if not "(" in line:
            continue
        if not "::" in line:
            continue
        funname = ''
        if ':' in line:
            funname = line.strip()[line.find(':') + 2:line.find('(') - 0] # ::OnHint
            # print funname
        else:
            funname = line.strip()[1:line.find('(') - 0] # void__fastcallTMain::OnHint
            # print funname


        filename =line.strip()[line.find('@@@') + 4: -7] #CRHMmain.cpp_nocom
        # --adding the root node--
        #root opening
        if '<root>' in line:
            stack.append('root')
            G.add_nodes_from(['root'])

        #root closing
        elif '</root>' in line:
            stack.pop()

        #opening other than root
        elif '</' not in line:
            if view == 1:
                # stack.append(funname[2:len(funname)])
                stack.append(funname)
            else:
                stack.append(filename)

            parent = stack[len(stack)-2]
            child = stack[len(stack)-1]

            #print stack[len(stack)-2]
            #print len(stack)-2
            #print stack[-1]

            G.add_edges_from([(parent, child)])

        else:
            try:
                stack.pop()
            except Exception as e:
                pass

    #end of loop

    #print stack
    print('unique scenario extracted for', f.name)

    nx.draw(G, with_labels=True)
    plt.show()

    return

def splitWordAndMakeSentence(paths):

    sentencePath = []

    for p in paths:

        str = ''
        for w in p:
            splitted = re.sub('(?!^)([A-Z][a-z]+)', r' \1', w).split()

            for s in splitted:
                str += s + ' '

            print(str)
            sentencePath.append(str)
    print(sentencePath)
    return sentencePath

def crhm_analysis():
    f0 =open('build+run.log')
    buildgraph(f0, 1)
    #tgf_to_networkX()
    G.remove_edges_from(G.selfloop_edges())
    extracting_source_and_exit_node()
    #print(S)
    #print(T)
    extracting_execution_paths()
    #print(G.edges())
    #print(execution_paths)

    df = pd.DataFrame(splitWordAndMakeSentence(execution_paths))
    df.to_csv('people.csv')

    return


def python_analysis():
    tgf_to_networkX()
    G.remove_edges_from(G.selfloop_edges())
    extracting_source_and_exit_node()
    extracting_execution_paths()
    df = pd.DataFrame(splitWordAndMakeSentence(execution_paths))
    df.to_csv('people.csv')

    return

#python_analysis()
#print(extract_function_name('add_defines_edge\\n(/home/avijit/github/pyan/pyan/analyzer.py:1247)\n'))
crhm_analysis()
