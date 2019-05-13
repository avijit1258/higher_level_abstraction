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
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from collections import defaultdict

from scipy.spatial import distance as ssd

import queue

import math

from spacy.lang.en import English

import nltk

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

from gensim import corpora

import pickle

import gensim


from scipy.cluster.hierarchy import cut_tree



# caller_callee = [('F1', 'F3'), ('F1', 'F7'), ('F3', 'F7'), ('F4', 'F1')]
# self calling edge is omitted for now

# caller callee for generating
caller_callee = [('F0', 'F1'), ('F0', 'F2'), ('F0', 'F3'), ('F1', 'F3'), ('F2', 'F0'), ('F2', 'F1'), ('F4', 'F2'), ('F5', 'F1')]

S = []

T = []

text_data = []

nltk.download('wordnet')

parser = English()

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

execution_paths = []
G = nx.DiGraph()
function_id_to_name = {}
# G.add_edges_from(caller_callee)
# G = nx.drawing.nx_agraph.read_dot('/pyan/pyan/pyan.dot')
# G = nx.DiGraph(nx.drawing.nx_pydot.read_dot('pyan/pyan/pyan.dot'))
# G = nx.read_graphml('pyan/pyan/pyan_yed.txt')
# nx.draw(G, with_labels=True)
# plt.show()


def extracting_source_and_exit_node():
    print('In degree')
    for s, v in G.in_degree():
        # print(s, v)
        if v == 0:
            S.append(s)
            # print(s)
    print(len(S))
    print('Out degree')
    for t, v in G.out_degree():
        # print(t, v)
        if v == 0:
            T.append(t)
            # print(t)

    print(len(T))

    return


def extract_function_name(str):

    end = str.find('\\')

    return str[:end]


def tgf_to_networkX():

        f = open("detectron_tgf.txt", "r")

        graph_started = False
        for line in f:

            if line.find('#') != -1:
                #print(line.find('#'))
                graph_started = True
                continue

            if graph_started == True:
                edge_info = line.split()
                # print(edge_info)
                # G.add_edge(function_id_to_name[edge_info[0]], function_id_to_name[edge_info[1]])
                G.add_edge(edge_info[0], edge_info[1])

            if graph_started == False:
               ln = line.split(' ')
               #print(ln)
               function_id_to_name[ln[0]] = extract_function_name(ln[1])

        nx.draw(G, with_labels=True)
        plt.savefig('call-graph.png')
        plt.show()

        return


def extracting_execution_paths():

    for s in S:
        for t in T:
            # print(list(nx.dfs_preorder_nodes(G, s)))
            # print(list(nx.all_simple_paths(G, s, t)))
            # execution_paths.append(list(nx.all_simple_paths(G, s, t)))
            unpack_path = list(nx.all_simple_paths(G, s, t))
            for p in unpack_path:
                execution_paths.append(p)


def buildgraph(f, view):
    # g = nx.DiGraph()
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
            funname = line.strip()[1:line.find('(') - 0]  # void__fastcallTMain::OnHint
            # print funname

        filename =line.strip()[line.find('@@@') + 4: -7]  # CRHMmain.cpp_nocom
        # --adding the root node--
        # root opening
        if '<root>' in line:
            stack.append('root')
            G.add_nodes_from(['root'])

        # root closing
        elif '</root>' in line:
            stack.pop()

        # opening other than root
        elif '</' not in line:
            if view == 1:
                # stack.append(funname[2:len(funname)])
                stack.append(funname)
            else:
                stack.append(filename)

            parent = stack[len(stack)-2]
            child = stack[len(stack)-1]

            # print stack[len(stack)-2]
            # print len(stack)-2
            # print stack[-1]

            G.add_edges_from([(parent, child)])

        else:
            try:
                stack.pop()
            except Exception as e:
                pass

    # end of loop

    # print stack
    print('unique scenario extracted for', f.name)

    nx.draw(G, with_labels=True)
    plt.savefig('crhm.png')
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
    f0 =open('clogs.txt')
    buildgraph(f0, 1)
    #tgf_to_networkX()
    G.remove_edges_from(G.selfloop_edges())
    extracting_source_and_exit_node()
    #print(S)
    #print(T)
    extracting_execution_paths()
    #print(G.edges())
    #print(execution_paths)

    mat = jaccard_distance_matrix(execution_paths)

    clustering_using_scipy(mat)
    # clustering_using_sklearn(mat)

    plt.show()

    # Storing execution paths for use in topic_modelling.py
    # df = pd.DataFrame(splitWordAndMakeSentence(execution_paths))
    # df.to_csv('people.csv')

    return


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    # print(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return 1 - float(intersection / union)


def jaccard_distance_matrix(paths):

    length = len(paths)
    Matrix = [[0 for x in range(length)] for y in range(length)]
    for i in range(len(paths)):
        for j in range(len(paths)):
            Matrix[i][j] = jaccard_similarity(paths[i], paths[j])

    return Matrix


def plot_dendrogram(model, mt,**kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # This block of code is for drawing single linkage algorithm for
    # Z = linkage(mt, 'single')
    # fig = plt.figure(figsize=(25, 10))
    # dn = dendrogram(Z)

    # Plot the corresponding dendrogram
    d = dendrogram(linkage_matrix, **kwargs, p = 30, truncate_mode='lastp')
    #print(d)


def clustering_using_scipy(mt):

    print('Execution paths : ', len(execution_paths))
    #print(mt)
    # npa = np.asarray(execution_paths)
    # Y = pdist(npa, 'jaccard')
    Z = linkage(ssd.squareform(mt), 'ward')
    # print(Z)
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(Z, truncate_mode='lastp', p=200)
    rootnode, nodelist = to_tree(Z, rd=True)

    nodes = bfs(nodelist, rootnode.id, 7)

    # labels = bfs_leaf_node(nodelist, 6729)
    # print(labels)

    for i in nodes:
        print(i)
        labels = bfs_leaf_node(nodelist, i)
        print('--------------#######--------')
        print('Cluster:', i, 'Count:', nodelist[i].count)
        # tf_idf_score_for_scipy_cluster(labels)
        # print('topic modelling label')
        # topic_model(labels)
        print('-------------#######-------')
        print(i)

    # print(rootnode.id)
    # label = []
    # for i in nodelist:
    #     # print(i.id)
    #     # node = nodelist[i.id]
    #     # print(node.left.id)
    #     # print(i.count)
    #     # print(i.left)
    #     # print(i.right)
    #     if i.count == 1:
    #         label.append(i.id)

    cluster_view(Z, dn)

    # plt.show()

    return


def bfs_leaf_node(nodelist, id):

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
        # print(q.qsize())
        p = q.get()
        # print(p)
        # print(q.qsize())
        count = count + 1

        # print(p, ' ', nodelist[p].count)
        visited[p] = 1

        if nodelist[p].count == 1:
            leaf_nodes.append(p)
            continue

        if visited[nodelist[p].left.id] == 0:
            q.put(nodelist[p].left.id)
        if visited[nodelist[p].right.id] == 0:
            q.put(nodelist[p].right.id)

        if count == nodelist[id].count:
            break

    return leaf_nodes


def bfs(nodelist, id, depth):

    # node = nodelist[id]
    nodes = []
    count = 0
    visited = [0] * len(nodelist)
    q = queue.Queue()
    q.put(id)
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
            print(p, ' ', nodelist[p].count)
            visited[p] = 1
            continue

        if visited[nodelist[p].left.id] == 0:
            q.put(nodelist[p].left.id)
        if visited[nodelist[p].right.id] == 0:
            q.put(nodelist[p].right.id)

        nodes.append(p)
        print(p, ' ', nodelist[p].count)
        visited[p] = 1

        if math.ceil(math.log(count + 1, 2)) == depth:
            break

    return nodes


def execution_path_to_sentence(labels):

    documents = []

    for l in labels:
        str = ''
        for e in execution_paths[l]:
            str += e
            str += ' '
        documents.append(str)

    return documents


def tf_idf_score_for_scipy_cluster(labels):

    # print(execution_paths[labels[0]])
    # print(labels)

    # txt1 = ['His smile was not perfect', 'His smile was not not not not perfect', 'she not sang']
    txt1 = execution_path_to_sentence(labels)
    tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
    txt_fitted = tf.fit(txt1)
    txt_transformed = txt_fitted.transform(txt1)

    # print(tf.vocabulary_)
    #
    feature_names = np.array(tf.get_feature_names())
    # sorted_by_idf = np.argsort(tf.idf_)
    # print("Features with lowest idf:\n{}".format(
    #     feature_names[sorted_by_idf[:5]]))
    # print("\nFeatures with highest idf:\n{}".format(
    #     feature_names[sorted_by_idf[-5:]]))

    max_val = txt_transformed.max(axis=0).toarray().ravel()

    # sort weights from smallest to biggest and extract their indices
    # print(max_val)
    sort_by_tfidf = max_val.argsort()

    # print("Features with lowest tfidf:\n{}".format(
    #     max_val[sort_by_tfidf[:5]]))

    print("\nFeatures with highest tfidf: \n{}".format(
        max_val[sort_by_tfidf[-5:]]))

    # print("Features with lowest tfidf:\n{}".format(
    #     feature_names[sort_by_tfidf[:5]]))

    print("\nFeatures with highest tfidf: \n{}".format(
        feature_names[sort_by_tfidf[-5:]]))

    return


def tf_idf_score(labels):

    clusters = defaultdict(list)
    flat_list = defaultdict(list)
    # print(labels)
    # print(len(labels))
    for i in range(len(labels)):
        clusters[labels[i]].append(execution_paths[i])

    for k, v in clusters.items():
        flat_list[k] = [item for sublist in clusters[k] for item in sublist]
        # vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', smooth_idf=True, use_idf=False)
        # print(np.array(execution_paths[c]).ravel())
        # print('clusters')
        # print(k)
        # X = vectorizer.fit_transform([item for sublist in clusters[k] for item in sublist])
        # print(vectorizer.idf_)
        # print(vectorizer.vocabulary_)
        # print(vectorizer.get_feature_names())
        #
        # c = 0
        # for d in vectorizer.get_feature_names():
        #     c += 1
        #     print(function_id_to_name[d])
        #     if c == 5:
        #         break
    print(flat_list)
    vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', smooth_idf=True, use_idf=False)
    X = vectorizer.fit_transform(flat_list)
    print(X)

    return


def clustering_using_sklearn(mat):

    model = AgglomerativeClustering(affinity='precomputed', n_clusters=10, linkage='single').fit(mat)
    print(model.labels_)
    plot_dendrogram(model, mat, labels=model.labels_)
    tf_idf_score(model.labels_)

    return


def python_analysis():
    tgf_to_networkX()
    G.remove_edges_from(G.selfloop_edges())
    extracting_source_and_exit_node()
    extracting_execution_paths()

    # df = pd.DataFrame(splitWordAndMakeSentence(execution_paths)) This line is for extracting words from function name which will be necessary for topic modeling application

    # exporting execution paths to be used in topic modeling
    # df = pd.DataFrame(execution_paths)
    # df.to_csv('people.csv')

    mat = jaccard_distance_matrix(execution_paths)

    clustering_using_scipy(mat)
    # clustering_using_sklearn(mat)

    plt.show()

    return


def cluster_view(Z, dend):
    X = flatten(dend['icoord'])
    Y = flatten(dend['dcoord'])
    leave_coords = [(x, y) for x, y in zip(X, Y) if y == 0]

    # in the dendogram data structure,
    # leave ids are listed in ascending order according to their x-coordinate
    order = np.argsort([x for x, y in leave_coords])
    id_to_coord = dict(zip(dend['leaves'], [leave_coords[idx] for idx in order]))  # <- main data structure

    # ----------------------------------------
    # get coordinates of other nodes

    # this should work but doesn't:

    # # traverse tree from leaves upwards and populate mapping ID -> (x,y);
    # # use linkage matrix to traverse the tree optimally
    # # (each row in the linkage matrix corresponds to a row in dend['icoord'] and dend['dcoord'])
    # root_node, node_list = to_tree(Z, rd=True)
    # for ii, (X, Y) in enumerate(zip(dend['icoord'], dend['dcoord'])):
    #     x = (X[1] + X[2]) / 2
    #     y = Y[1] # or Y[2]
    #     node_id = ii + len(dend['leaves'])
    #     id_to_coord[node_id] = (x, y)

    # so we need to do it the hard way:

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
    print(ids_left)

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


def flatten(l):
    return [item for sublist in l for item in sublist]


def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


def prepare_text_for_lda(text):
    tokens = tokenize(text)
    #print(tokens)
    tokens = [token for token in tokens if len(token) >= 2]
    #print(tokens)
    tokens = [token for token in tokens if token not in en_stop]
    #print(tokens)
    tokens = [get_lemma(token) for token in tokens]
    #print(tokens)
    return tokens


def topic_model(labels):

    txt = execution_path_to_sentence(labels)

    for line in txt:
        # print(line)
        tokens = prepare_text_for_lda(line)
        # if random.random() > .99:
        # print(tokens)
        text_data.append(tokens)

    # print(text_data)
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]

    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')

    NUM_TOPICS = 5
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=5)
    for topic in topics:
        print(topic)


# python_analysis()

# print(extract_function_name('add_defines_edge\\n(/home/avijit/github/pyan/pyan/analyzer.py:1247)\n'))


crhm_analysis()
