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

import xlsxwriter
from timeit import default_timer as timer
import multiprocessing

from xlsxwriter import worksheet


workbook = xlsxwriter.Workbook('sklearn.xlsx')
worksheet = workbook.add_worksheet()

class ClusteringCallGraph:
    S = []

    T = []

    tree = []

    text_data = []

    subject_system = ''

    nltk.download('wordnet')

    special_functions = ['lambda', 'genexpr', 'listcomp', 'setcomp', 'dictcomp']

    parser = English()

    nltk.download('stopwords')
    en_stop = set(nltk.corpus.stopwords.words('english'))

    execution_paths = []
    G = nx.DiGraph()
    function_id_to_name = {}

    row = 1

    count = 0



    # worksheet.write('ClusterId', 'Execution_Paths', 'Naming_using_our_approach')
    worksheet.write(0,0, 'Cluster Id')
    worksheet.write(0,1, 'Execution_Paths')
    worksheet.write(0,2, 'tfidf_word')
    worksheet.write(0, 3, 'tfidf_method')
    worksheet.write(0, 4, 'lda_word')
    worksheet.write(0, 5, 'lda_method')
    worksheet.write(0, 6, 'lsi_word')
    worksheet.write(0, 7, 'lsi_method')

    def __del__(self):

        print('deleted')

    def python_analysis(self):
        self.tgf_to_networkX()
        self.G.remove_edges_from(self.G.selfloop_edges())
        self.extracting_source_and_exit_node()
        start = timer()
        self.extracting_execution_paths()
        end = timer()
        print('Time required for extracting_execution_paths: ', end - start)
        # df = pd.DataFrame(splitWordAndMakeSentence(execution_paths)) This line is for extracting words from function name which will be necessary for topic modeling application

        # exporting execution paths to be used in topic modeling
        # df = pd.DataFrame(execution_paths)
        # df.to_csv('people.csv')
        start = timer()
        mat = self.jaccard_distance_matrix(self.execution_paths)
        end = timer()
        print('Time required for jaccard_distance_matrix: ', end - start)
        # clustering_using_sklearn(mat)

        # plt.show()
        self.G.clear()

        return self.clustering_using_scipy(mat)

    def tgf_to_networkX(self):
        self.subject_system = input('Enter name of the subject system: \n')
        print('thanks a lot')
        # path = easygui.fileopenbox()
        f = open(self.subject_system, "r")
        # f = open(path, "r")
        G = nx.DiGraph()
        # print("Function name: ")
        graph_started = False
        for line in f:

            if line.find('#') != -1:
                # print(line.find('#'))
                graph_started = True
                continue

            if graph_started == True:
                edge_info = line.split()
                # print(edge_info)
                # G.add_edge(function_id_to_name[edge_info[0]], function_id_to_name[edge_info[1]])
                if self.function_id_to_name[edge_info[0]] in self.special_functions or self.function_id_to_name[edge_info[1]] in self.special_functions:
                    continue
                self.G.add_edge(edge_info[0], edge_info[1])

            if graph_started == False :
                ln = line.split(' ')
                # print(ln)
                self.function_id_to_name[ln[0]] = self.extract_function_name(ln[1])
                # print(ln[0])

        nx.draw(self.G, with_labels=True)
        plt.savefig('call-graph.png')
        plt.show()

        return

    def extracting_source_and_exit_node(self):
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
        print('Extracting execution paths')
        for s in self.S:
            for t in self.T:
                # print(list(nx.dfs_preorder_nodes(G, s)))
                # print(list(nx.all_simple_paths(G, s, t)))
                # execution_paths.append(list(nx.all_simple_paths(G, s, t)))
                unpack_path = list(nx.all_simple_paths(self.G, s, t))
                for p in unpack_path:
                    self.execution_paths.append(p)

    def jaccard_distance_matrix(self, paths):
        print('jaccard_distance_matrix')
        length = len(paths)
        Matrix = [[0 for x in range(length)] for y in range(length)]
        for i in range(len(paths)):
            for j in range(len(paths)):
                Matrix[i][j] = self.jaccard_similarity(paths[i], paths[j])

        return Matrix

    def labeling_cluster(self, execution_paths_of_a_cluster, k, v):
        # print(k,'blank document', execution_paths_of_a_cluster)
        # print('Here we go',self.execution_path_to_sentence(execution_paths_of_a_cluster))

        tfidf_method = self.tf_idf_score_for_scipy_cluster(execution_paths_of_a_cluster, 'method')
        tfidf_word = self.tf_idf_score_for_scipy_cluster(execution_paths_of_a_cluster, 'word')
        lda_method = self.topic_model_lda(execution_paths_of_a_cluster, 'method')
        lda_word = self.topic_model_lda(execution_paths_of_a_cluster, 'word')
        lsi_method = self.topic_model_lda(execution_paths_of_a_cluster, 'method')
        lsi_word = self.topic_model_lda(execution_paths_of_a_cluster, 'word')

        worksheet.write(self.row, 0, k)
        worksheet.write(self.row, 1, self.execution_path_to_sentence(execution_paths_of_a_cluster))
        worksheet.write(self.row, 2, self.merge_words_as_sentence(tfidf_word))
        worksheet.write(self.row, 3, self.id_to_sentence(tfidf_method))
        worksheet.write(self.row, 4, lda_word)
        worksheet.write(self.row, 5, lda_method)
        worksheet.write(self.row, 6, lsi_word)
        worksheet.write(self.row, 7, lsi_method)

        # tf = self.tf_idf_score_for_scipy_cluster(execution_paths_of_a_cluster)
        # tm = self.topic_model_lda(execution_paths_of_a_cluster)
        # tm = self.topic_model_lsi(execution_paths_of_a_cluster)
        # print(self.count,'' ,tf)
        # worksheet.write(self.row, 0, k)
        # worksheet.write(self.row, 1, self.execution_path_to_sentence(execution_paths_of_a_cluster))
        # worksheet.write(self.row, 2, self.merge_words_as_sentence(tf)) # split_method_tfidf
        # worksheet.write(self.row, 2, self.id_to_sentence(tfidf_method)) # method tfidf
        # worksheet.write(self.row, 2, tm)
        self.row += 1
        # worksheet.write(k, self.execution_path_to_sentence(execution_paths_of_a_cluster), tf)

        # print('topic modelling label')
        # tm = self.topic_model(labels)
        # print('-------------#######-------')
        # Considering functions names as unit
        # self.tree.append({'key': k, 'parent': v, 'tf_name': self.id_to_sentence(tf), 'tm_name': 'Hello topic'})
        # self.tree.append({'key': k, 'parent': v, 'tf_name': 'Hello tfidf', 'tm_name': tm})
        # Considering words in functions name as unit
        # self.tree.append({'key': k, 'parent': v, 'tf_name': self.merge_words_as_sentence(tf), 'tm_name': 'Hello topic'})
        self.tree.append({'key': k, 'parent': v, 'tfidf_word': self.merge_words_as_sentence(tfidf_word), 'tfidf_method': self.id_to_sentence(tfidf_method), 'lda_word': lda_word, 'lda_method': lda_method, 'lsi_word': lsi_word, 'lsi_method': lsi_method})
        return

    def clustering_using_scipy(self, mt):

        # print('Execution paths : ', len(self.execution_paths))
        # print(mt)
        # npa = np.asarray(execution_paths)
        # Y = pdist(npa, 'jaccard')
        start = timer()
        Z = linkage(ssd.squareform(mt), 'ward')
        # print(Z)
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z, truncate_mode='lastp', p=200)
        rootnode, nodelist = to_tree(Z, rd=True)

        nodes = self.bfs(nodelist, rootnode.id, math.ceil(math.log(len(nodelist) + 1, 2)))
        nodes_with_parent = self.bfs_with_parent(nodelist, rootnode.id, math.ceil(math.log(len(nodelist) + 1, 2)))
        end = timer()
        print('Time required for clustering: ', end - start)
        # labels = bfs_leaf_node(nodelist, 6729)
        # print(labels)

        start = timer()
        for k,v in nodes_with_parent.items():
            if nodelist[k].count == 1:
                continue
            execution_paths_of_a_cluster = self.bfs_leaf_node(nodelist, k)
            # print(k, 'cluster using scipy', labels)
            # p = multiprocessing.Process(target=self.labeling_cluster,args=(labels,k,v,))
            # p.start()
            self.count += 1
            if self.count == 300:
                print('Hello')
                break
            self.labeling_cluster(execution_paths_of_a_cluster, k, v)

            #print('--------------#######--------')
            # print('Cluster:', k, 'Count:', nodelist[k].count)
            # tf = self.tf_idf_score_for_scipy_cluster(labels)
            # print('topic modelling label')
            # tm = self.topic_model(labels)
            # print('-------------#######-------')
            # tree.append({'key':k, 'parent': v, 'tf_name': tf, 'tm_name': tm})
        end = timer()
        print('Time required for labeling using 6 techniques', end - start)
        # for i in nodes:
        #     print(i)
        #     labels = self.bfs_leaf_node(nodelist, i)
        #     print('--------------#######--------')
        #     print('Cluster:', i, 'Count:', nodelist[i].count)
        #     self.tf_idf_score_for_scipy_cluster(labels)
        #     print('topic modelling label')
        #     self.topic_model(labels)
        #     print('-------------#######-------')
        #     print(i)

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

        # self.cluster_view(Z, dn)

        # plt.show()
        print(self.tree, file=open('tree'+self.subject_system, 'w'))

        return self.tree

    def extract_function_name(self,str):
        end = str.find('\\')

        return str[:end]

    def jaccard_similarity(self, list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        # print(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return 1 - float(intersection / union)

    def bfs_leaf_node(self, nodelist, id):

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

    def tf_idf_score_for_scipy_cluster(self, clusters, method_or_word):

        # print(execution_paths[labels[0]])
        # print('tf_idf_score_for_scipy_cluster')
        # print(labels)

        txt1 = ['His smile was not perfect', 'His smile was not not not not perfect', 'she not sang']
        try:


            # for i in labels:
            #     print(self.execution_paths[i])
            if method_or_word == 'method':
                txt1 = self.make_documents_for_a_cluster_tfidf_method(clusters)
            elif method_or_word == 'word':
                txt1 = self.make_documents_for_a_cluster_tfidf_word(clusters)
            # txt1 = self.make_documents_for_a_cluster_tfidf_word(clusters)
            # txt1 = self.make_documents_for_a_cluster_tfidf_method(clusters)
            # print('Txt1: ', txt1, 'Clusters:',clusters)
            # when digits are passed as words(un-comment this when methods are used as unit)
            # tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word', token_pattern='\d+')
            # when words are used means strings
            tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word', token_pattern='(?u)\\b\\w\\w+\\b')
            txt_fitted = tf.fit(txt1)
            txt_transformed = txt_fitted.transform(txt1)
        except:
            print('Here I got you',clusters, 'In a sentence:', txt1)
            

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

        # print("\nFeatures with highest tfidf: \n{}".format(
        #     max_val[sort_by_tfidf[-5:]]))

        # ~ printing highest frequency
        # print("Features with lowest tfidf:\n{}".format(
        #     feature_names[sort_by_tfidf[:5]]))

        # print("\nFeatures with highest tfidf: \n{}".format(
        #     feature_names[sort_by_tfidf[-5:]]))

        return feature_names[sort_by_tfidf[-5:]]

    def make_documents_for_a_cluster_tfidf_method(self, clusters):

        documents = []

        for c in clusters:
            str = ''
            for e in self.execution_paths[c]:
                str += e
                str += ' '
            documents.append(str)


        return documents

    def make_documents_for_a_cluster_tm_method(self, clusters):

        documents = []

        for c in clusters:
            str = ''
            for e in self.execution_paths[c]:
                str += self.function_id_to_name[e]
                str += ' '
            documents.append(str)


        return documents

    def make_documents_for_a_cluster_tfidf_word(self, clusters):

        documents = []

        for c in clusters:
            str = ''
            for e in self.execution_paths[c]:
                # print(self.merge_words_as_sentence(self.function_id_to_name[e].split("_")))
                str += self.merge_words_as_sentence(self.function_id_to_name[e].split("_"))
                str += ' '
            # print('\n')
            documents.append(str)

        return documents
    def make_documents_for_a_cluster_tm_word(self, clusters):

        documents = []

        for c in clusters:
            str = ''
            for e in self.execution_paths[c]:
                # print(self.merge_words_as_sentence(self.function_id_to_name[e].split("_")))
                str += self.merge_words_as_sentence(self.function_id_to_name[e].split("_"))
                str += ' '
            # print('\n')
            documents.append(str)

        return documents

    def merge_words_as_sentence(self, identifiers):

        result = []
        st = ''
        # to omit empty words
        for i in identifiers:
            if i == '':
                continue
            else:
                result.append(i)

        for r in result:
            st += r
            st += ' '

        return st

    def topic_model_lda(self, labels, method_or_word):

        if method_or_word == 'method':
            txt = self.make_documents_for_a_cluster_tm_method(labels)
        elif method_or_word == 'word':
            txt = self.make_documents_for_a_cluster_tm_word(labels)

        # txt = self.make_documents_for_a_cluster_tm_method(labels)
        # txt = self.make_documents_for_a_cluster_tm_word(labels)

        for line in txt:
            # print(line)
            tokens = self.prepare_text_for_lda(line)
            # if random.random() > .99:
            # print(tokens)
            self.text_data.append(tokens)

        # print(text_data)
        dictionary = corpora.Dictionary(self.text_data)
        corpus = [dictionary.doc2bow(text) for text in self.text_data]

        pickle.dump(corpus, open('corpus.pkl', 'wb'))
        dictionary.save('dictionary.gensim')

        NUM_TOPICS = 5
        ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=3)
        # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=3)
        ldamodel.save('model5.gensim')
        #topics = ldamodel.print_topics(num_words=5)
        # for topic in topics:
        #    print(topic)
        topics = ldamodel.print_topic(0,topn=5)

        return topics

    def topic_model_lsi(self, labels, method_or_word):

        if method_or_word == 'method':
            txt = self.make_documents_for_a_cluster_tm_method(labels)
        elif method_or_word == 'word':
            txt = self.make_documents_for_a_cluster_tm_word(labels)

        # txt = self.make_documents_for_a_cluster_tm_method(labels)
        # txt = self.make_documents_for_a_cluster_tm_word(labels)
        print(txt)

        for line in txt:
            # print(line)
            tokens = self.prepare_text_for_lda(line)
            # if random.random() > .99:
            # print(tokens)
            self.text_data.append(tokens)

        # print(text_data)
        dictionary = corpora.Dictionary(self.text_data)
        corpus = [dictionary.doc2bow(text) for text in self.text_data]

        pickle.dump(corpus, open('corpus.pkl', 'wb'))
        dictionary.save('dictionary.gensim')

        NUM_TOPICS = 5
        # ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=3)
        # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=3)
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics=5, id2word=dictionary)
        lsimodel.save('model5.gensim')
        # topics = ldamodel.print_topics(num_words=5)
        # for topic in topics:
        #    print(topic)
        topics = lsimodel.print_topic(0, topn=5)

        return topics

    def prepare_text_for_lda(self, text):
        tokens = self.tokenize(text)
        # print(tokens)
        tokens = [token for token in tokens if len(token) >= 2]
        # print(tokens)
        tokens = [token for token in tokens if token not in self.en_stop]
        # print(tokens)
        tokens = [self.get_lemma(token) for token in tokens]
        # print(tokens)
        return tokens

    def tokenize(self, text):
        lda_tokens = []
        tokens = self.parser(text)
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

    def get_lemma(self, word):
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma

    def cluster_view(self, Z, dend):
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
        return [item for sublist in l for item in sublist]

    def bfs(self, nodelist, id, depth):

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
                # print(p, ' ', nodelist[p].count)
                visited[p] = 1
                continue

            if visited[nodelist[p].left.id] == 0:
                tree[nodelist[p].left.id] = nodelist[p].id
                q.put(nodelist[p].left.id)
            if visited[nodelist[p].right.id] == 0:
                tree[nodelist[p].right.id] = nodelist[p].id
                q.put(nodelist[p].right.id)

            nodes.append(p)
            # print(p, ' ', nodelist[p].count)
            visited[p] = 1

            if math.ceil(math.log(count + 1, 2)) == depth:
                break
        # print('bfs_with_parent')
        # print(tree)
        return tree

        return

    def id_to_sentence(self,execution_paths):

        str = ''

        for l in execution_paths:
            str += self.function_id_to_name[l]
            str += ' '

        return str

    def execution_path_to_sentence(self, execution_paths_of_a_cluster):

        documents = []

        try:
            str = ''
            for l in execution_paths_of_a_cluster:

                for e in self.execution_paths[l]:
                    str += self.function_id_to_name[e]
                    str += ', '
                str += ' ;'
                # documents.append(str)
        except:
            print('Crushed : ', e)

        return str


c = ClusteringCallGraph()

c.python_analysis()

workbook.close()

