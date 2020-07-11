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
from gensim.summarization.summarizer import summarize
from gensim.summarization.textcleaner import clean_text_by_sentences as _clean_text_by_sentences
from gensim.summarization.textcleaner import get_sentences
import xlsxwriter
from timeit import default_timer as timer
import multiprocessing
from xlsxwriter import worksheet
from prefixspan import PrefixSpan

from PlayingWithAST import *



workbook = xlsxwriter.Workbook('pyan.xlsx')
worksheet = workbook.add_worksheet()

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

    pwa = PlayingWithAST()

    function_name_to_docstring = pwa.get_all_method_docstring_pair_of_a_project('/home/avb307/projects/hla_dataset/pyan')


    # worksheet.write('ClusterId', 'Execution_Paths', 'Naming_using_our_approach')
    worksheet.write(0,0, 'Cluster Id')
    worksheet.write(0,1, 'Execution_Paths')
    worksheet.write(0,2, 'tfidf_word')
    worksheet.write(0, 3, 'tfidf_method')
    worksheet.write(0, 4, 'lda_word')
    worksheet.write(0, 5, 'lda_method')
    worksheet.write(0, 6, 'lsi_word')
    worksheet.write(0, 7, 'lsi_method')
    worksheet.write(0, 8, 'text_summary')
    worksheet.write(0, 9, 'SPM method')


    def __del__(self):
        """ deletes the ClusteringCallGraph class objects """
        print('deleted')

    def python_analysis(self):
        """ analyzing python programs to build cluster tree of execution paths. """
        self.tgf_to_networkX()
        self.G.remove_edges_from(self.G.selfloop_edges())
        self.extracting_source_and_exit_node()
        start = timer()
        self.extracting_execution_paths()
        end = timer()
        print('Time required for extracting_execution_paths: ', end - start)
        print('No. of execution paths', len(self.execution_paths))

        self.remove_redundant_ep()
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

        execution_paths.sort(key = len, reverse = True)

        redundant_ep = []
        for e in self.execution_paths:
            if e in redundant_ep:
                continue
            for f in self.execution_paths:
                if e != f:
                    print(e, f)
                    if self.check_ep_overlap_from_start(e, f):
                        redundant_ep.append(f)
        for r in redundant_ep:
            execution_paths.remove(r)
            
        print(' Filtered execution path length ', len(self.execution_paths))


    def tgf_to_networkX(self):
        """ converting tgf file to a networkX graph"""
        self.subject_system = input('Enter name of the subject system: \n')
        # self.subject_system = '/home/avb307/projects/higher_level_abstraction/tool/calculator.txt'
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
            for t in self.T:
                # print(list(nx.dfs_preorder_nodes(G, s)))
                # print(list(nx.all_simple_paths(G, s, t)))
                # execution_paths.append(list(nx.all_simple_paths(G, s, t)))
                unpack_path = list(nx.all_simple_paths(self.G, s, t))
                for p in unpack_path:
                    self.execution_paths.append(p)

    def jaccard_distance_matrix(self, paths):
        """ creating distance matrix using jaccard similarity value """
        print('jaccard_distance_matrix')
        length = len(paths)
        Matrix = [[0 for x in range(length)] for y in range(length)]
        for i in range(len(paths)):
            for j in range(len(paths)):
                Matrix[i][j] = self.jaccard_similarity(paths[i], paths[j])
        
        # print('Paths :', paths)
        # print('Matrix :', Matrix)

        return Matrix

    def labeling_cluster(self, execution_paths_of_a_cluster, k, v):
        """ Labelling a cluster using six variants """
        # print(k,'blank document', execution_paths_of_a_cluster)
        # print('Here we go',self.execution_path_to_sentence(execution_paths_of_a_cluster))

        
        spm_method = self.mining_sequential_patterns(execution_paths_of_a_cluster)
        tfidf_method = self.tf_idf_score_for_scipy_cluster(execution_paths_of_a_cluster, 'method') 
        tfidf_word = 'IN: '+ str(k) + ', Name: ' + self.tf_idf_score_for_scipy_cluster(execution_paths_of_a_cluster, 'word') 
        lda_method = self.topic_model_lda(execution_paths_of_a_cluster, 'method')
        lda_word = self.topic_model_lda(execution_paths_of_a_cluster, 'word')
        lsi_method = self.topic_model_lsi(execution_paths_of_a_cluster, 'method')
        lsi_word = self.topic_model_lsi(execution_paths_of_a_cluster, 'word')
        text_summary = self.summarize_clusters_using_docstring(execution_paths_of_a_cluster, self.function_name_to_docstring)

        worksheet.write(self.row, 0, k)
        worksheet.write(self.row, 1, self.execution_path_to_sentence(execution_paths_of_a_cluster))
        worksheet.write(self.row, 2, tfidf_word)
        worksheet.write(self.row, 3, tfidf_method)
        worksheet.write(self.row, 4, lda_word)
        worksheet.write(self.row, 5, lda_method)
        worksheet.write(self.row, 6, lsi_word)
        worksheet.write(self.row, 7, lsi_method)
        worksheet.write(self.row, 8, text_summary)
        worksheet.write(self.row, 9, spm_method)
        self.row += 1
        
        self.tree.append({'key': k, 'parent': v, 'tfidf_word': tfidf_word, 'tfidf_method': tfidf_method, 'lda_word': lda_word, 'lda_method': lda_method, 'lsi_word': lsi_word, 'lsi_method': lsi_method, 'spm_method' : spm_method , 'text_summary': text_summary})
        return

    def clustering_using_scipy(self, mt):
        """ clustering execution paths using scipy """

        # print('Execution paths : ', len(self.execution_paths))
        # print(mt)
        # npa = np.asarray(execution_paths)
        # Y = pdist(npa, 'jaccard')
        start = timer()
        Z = linkage(ssd.squareform(mt), 'ward')
        # print('Z is here', Z)
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z, truncate_mode='lastp', p=200)
        rootnode, nodelist = to_tree(Z, rd=True)

        nodes = self.bfs(nodelist, rootnode.id, math.ceil(math.log(len(nodelist) + 1, 2)))
        nodes_with_parent = self.bfs_with_parent(nodelist, rootnode.id, math.ceil(math.log(len(nodelist) + 1, 2)))
        # print(nodes_with_parent)
        end = timer()
        print('Time required for clustering: ', end - start)
        # labels = bfs_leaf_node(nodelist, 6729)
        # print(labels)

        start = timer()
        for k,v in nodes_with_parent.items():
            if nodelist[k].count == 1:
                self.tree.append({'key': k, 'parent': v, 'tfidf_word': 'EP: '+ str(k) + ', Name: ' +self.pretty_print_leaf_node(self.execution_paths[k]), 'tfidf_method': '', 'lda_word': '', 'lda_method': '', 'lsi_word': '', 'lsi_method': '', 'spm_method': '','text_summary': 'hello summary'})
                continue
            execution_paths_of_a_cluster = self.bfs_leaf_node(nodelist, k)
            # print(k, 'Nodes leaf nodes are: ', execution_paths_of_a_cluster)
            # print(k, 'cluster using scipy', labels)
            # p = multiprocessing.Process(target=self.labeling_cluster,args=(labels,k,v,))
            # p.start()
            self.count += 1
            print('Cluster no: ',self.count)
            # if self.count == 300:
            #     print('Hello')
            #     break
            self.labeling_cluster(execution_paths_of_a_cluster, k, v)

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
        # print(self.tree, file=open('tree_calculator.txt', 'w'))
        return self.tree

    def extract_function_name(self,str):
        """ extracting function names from TGF file """
        end = str.find('\\')

        return str[:end]

    def jaccard_similarity(self, list1, list2):
        """ calculating jaccard similarity """
        intersection = len(list(set(list1).intersection(list2)))
        # print(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return 1 - float(intersection / union)

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

            # if count == nodelist[id].count:
            #     break

        return leaf_nodes

    def tf_idf_score_for_scipy_cluster(self, clusters, method_or_word):
        """ 
        Tfidf score calculation for a scipy cluster.
        """
        # print(execution_paths[labels[0]])
        # print('tf_idf_score_for_scipy_cluster')
        # print(labels)

        txt1 = ['His smile was not perfect', 'His smile was not not not not perfect', 'she not sang']
        try:

            if method_or_word == 'method':
                txt1 = self.make_documents_for_a_cluster_tfidf_method(clusters)
            elif method_or_word == 'word':
                txt1 = self.make_documents_for_a_cluster_tfidf_word(clusters)
                
            
            tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word', token_pattern='[a-zA-Z0-9]+')
        
            txt_transformed = tf.fit_transform(txt1)

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
        if method_or_word == 'method':
            return self.id_to_sentence(feature_names[sort_by_tfidf[-5:]])
        elif method_or_word == 'word':
            return self.merge_words_as_sentence(feature_names[sort_by_tfidf[-5:]])


    def make_documents_for_a_cluster_tfidf_method(self, clusters):
        """ 
        Making documents using execution paths of a cluster for tfidf method variant.
        """
        documents = []

        for c in clusters:
            str = ''
            for e in self.execution_paths[c]:
                str += e
                str += ' '
            documents.append(str)


        return documents

    def make_documents_for_a_cluster_tm_method(self, clusters):
        """
        Making documents using execution paths of a cluster for topic model method variant.
        """
        documents = []

        for c in clusters:
            str = ''
            for e in self.execution_paths[c]:
                str += self.function_id_to_name[e]
                str += ' '
            documents.append(str)


        return documents

    def make_documents_for_a_cluster_tfidf_word(self, clusters):
        """ 
        Making documents using execution paths of a cluster for tfidf word variant.
        """
        documents = []

        for c in clusters:
            str = ''
            for e in self.execution_paths[c]:
                # print(self.merge_words_as_sentence(self.function_id_to_name[e].split("_")))
                words_in_function_name = [w for w in self.function_id_to_name[e].split("_") if w not in self.en_stop]
                words_in_function_name = [ self.get_lemma(w) for w in words_in_function_name]
                str += self.merge_words_as_sentence(words_in_function_name)
                str += ' '
            # print('\n')
            documents.append(str)

        # print('make_documents_for_a_cluster_tfidf_word', documents)

        return documents
    def make_documents_for_a_cluster_tm_word(self, clusters):
        """ 
        Making documents using execution paths of a cluster for topic model word variant.
        """
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
        """ 
         Merging word as sentence.
        """
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

    def topic_model_output(self, topics):
        """ formatting topic model outputs """
        out = ' '

        for t in topics:
            out = out + t[0]
            out = out + ','

        return out

    def topic_model_lda(self, labels, method_or_word):
        """ 
        LDA algorithm for method and word variants.
        """
        self.text_data = []
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
        # topics = ldamodel.print_topic(0, topn=5)
        topics = ldamodel.show_topic(0, topn=5)
        topics = self.topic_model_output(topics)

        
        return topics

    def topic_model_lsi(self, labels, method_or_word):
        """ 
        LSI algorithm for both method and word variant.
        """

        self.text_data = []

        if method_or_word == 'method':
            txt = self.make_documents_for_a_cluster_tm_method(labels)
        elif method_or_word == 'word':
            txt = self.make_documents_for_a_cluster_tm_word(labels)

        # txt = self.make_documents_for_a_cluster_tm_method(labels)
        # txt = self.make_documents_for_a_cluster_tm_word(labels)
        # print(txt)

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
        # topics = lsimodel.print_topic(0, topn=5)

        topics = lsimodel.show_topic(0, topn=5)
        topics = self.topic_model_output(topics)

        return topics


    def prepare_text_for_lda(self, text):
        """ 
        Proprocessing text for LDA.
        """
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
        """ 
        Tokenize a word.
        """
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
        """ 
        Getting lemma of a word.
        """
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma

    def summarize_clusters_using_docstring(self, execution_paths_of_a_cluster, function_name_to_docstring):
        """  automatic text summarization for docstring of function names """
        
        text_for_summary = ''
        # count = 0
        for c in execution_paths_of_a_cluster:
            for f in self.execution_paths[c]:
                # print(self.function_id_to_name[f], ' ', function_name_to_docstring[self.function_id_to_name[f]])
                if self.function_id_to_name[f] in function_name_to_docstring:

                    if function_name_to_docstring[self.function_id_to_name[f]] is not None:
                        text_for_summary += function_name_to_docstring[self.function_id_to_name[f]] + ' '
                        # count += 1

        # print([self.execution_paths[e] for e in execution_paths_of_a_cluster])
        # print(text_for_summary)
        print(len(text_for_summary))
        # print(_clean_text_by_sentences(text_for_summary))
        # print(get_sentences(text_for_summary))
        # print(len(get_sentences(text_for_summary)))
        # count = 0
        # for sentence in get_sentences(text_for_summary):
        #     print(sentence)
        #     count += 1
        # if count <= 9:
        #     return 'Empty.'
        # if len(text_for_summary) <= 1:
        #     return 'Empty'

        try:
            return summarize(text_for_summary, word_count=25)
        except ValueError:
            return 'Empty'


    def mining_sequential_patterns(self, execution_paths_of_a_cluster):
        """ This function mines sequential patterns from execution paths """
        
        preprocess = [self.execution_paths[item] for item in execution_paths_of_a_cluster]
        
        ps = PrefixSpan(preprocess)

        top5 = ps.topk(5, closed = True)
        
        sentence = ''
        for i in top5:
            for j in i[1]:
                sentence += self.function_id_to_name[j] 
                if j != i[1][len(i[1])-1]:
                    sentence += ','
                
            sentence += ';\n'

        # print(sentence)
        return sentence

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

            # if math.ceil(math.log(count + 1, 2)) == depth:
            #     break
        # print('bfs_with_parent')
        # print(tree)
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
                str += '-->'

        return str

    def execution_path_to_sentence(self, execution_paths_of_a_cluster):
        """ 
        This function takes execution paths of a cluster. Then creates a printable string with execution paths with function names.
        """
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

