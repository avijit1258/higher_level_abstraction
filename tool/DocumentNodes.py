from xlsxwriter import worksheet
from prefixspan import PrefixSpan
import xlsxwriter
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
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class DocumentNodes:

    nltk.download('wordnet')
    parser = English()
    nltk.download('stopwords')
    en_stop = set(nltk.corpus.stopwords.words('english'))

    def __init__(self, output_directory, subject_system_name):
        self.workbook = xlsxwriter.Workbook(output_directory +subject_system_name+'.xlsx')
        self.worksheet = self.workbook.add_worksheet()
        self.row = 0
        self.initalize_sheet()
        self.execution_paths = []
        self.function_id_to_name = {}
        self.function_id_to_file_name = {}
        self.id_to_sentence = {}
        self.function_name_to_docstring = {}
        self.execution_path_to_sentence = {}

    def initalize_sheet(self):
        column = 0
        sheet_labels = ['Cluster Id', 'Execution_Paths', 'tfidf_word', 'tfidf_method', 'lda_word', 'lda_method', 'lsi_word', 'lsi_method', 'text_summary', 'SPM method']

        for column in range(len(sheet_labels)):
            self.worksheet.write(0, column, sheet_labels[column])

    def labeling_cluster(self, execution_paths_of_a_cluster, k, v):
        """ Labelling a cluster using six variants """
        # print(k,'blank document', execution_paths_of_a_cluster)
        # print('Here we go',self.execution_path_to_sentence(execution_paths_of_a_cluster))

        
        spm_method = self.mining_sequential_patterns(execution_paths_of_a_cluster)
        tfidf_method = self.tf_idf_score_for_scipy_cluster(execution_paths_of_a_cluster, 'method') 
        tfidf_word =  self.tf_idf_score_for_scipy_cluster(execution_paths_of_a_cluster, 'word') 
        lda_method = self.topic_model_lda(execution_paths_of_a_cluster, 'method')
        lda_word = self.topic_model_lda(execution_paths_of_a_cluster, 'word')
        lsi_method = self.topic_model_lsi(execution_paths_of_a_cluster, 'method')
        lsi_word = self.topic_model_lsi(execution_paths_of_a_cluster, 'word')
        text_summary = self.summarize_clusters_using_docstring(execution_paths_of_a_cluster, self.function_name_to_docstring)
        files_count, files = self.count_files_in_node(execution_paths_of_a_cluster)
        

        self.worksheet.write(self.row, 0, k)
        self.worksheet.write(self.row, 1, self.execution_path_to_sentence(execution_paths_of_a_cluster))
        self.worksheet.write(self.row, 2, tfidf_word)
        self.worksheet.write(self.row, 3, tfidf_method)
        self.worksheet.write(self.row, 4, lda_word)
        self.worksheet.write(self.row, 5, lda_method)
        self.worksheet.write(self.row, 6, lsi_word)
        self.worksheet.write(self.row, 7, lsi_method)
        self.worksheet.write(self.row, 8, text_summary)
        self.worksheet.write(self.row, 9, spm_method)
        self.row += 1
        
        return {'key': k, 'parent': v, 'tfidf_word': tfidf_word, 'tfidf_method': tfidf_method, 'lda_word': lda_word,\
                'lda_method': lda_method, 'lsi_word': lsi_word, 'lsi_method': lsi_method, 'spm_method' : spm_method ,\
                'text_summary': text_summary, 'files_count': files_count, 'files': files}

    def tf_idf_score_for_scipy_cluster(self, clusters, method_or_word):
        """ 
        Tfidf score calculation for a scipy cluster.
        """
    
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
        max_val = txt_transformed.max(axis=0).toarray().ravel()
        sort_by_tfidf = max_val.argsort()

        
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

       
        for line in txt:
            # print(line)
            tokens = self.prepare_text_for_lda(line)
            # if random.random() > .99:
            # print(tokens)
            self.text_data.append(tokens)

        # print(text_data)
        dictionary = corpora.Dictionary(self.text_data)
        corpus = [dictionary.doc2bow(text) for text in self.text_data]

        NUM_TOPICS = 5
        # ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=3)
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=3)
    
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


        for line in txt:
            # print(line)
            tokens = self.prepare_text_for_lda(line)
            # if random.random() > .99:
            # print(tokens)
            self.text_data.append(tokens)

        # print(text_data)
        dictionary = corpora.Dictionary(self.text_data)
        corpus = [dictionary.doc2bow(text) for text in self.text_data]

        # pickle.dump(corpus, open('corpus.pkl', 'wb'))
        # dictionary.save('dictionary.gensim')

        NUM_TOPICS = 5
        # ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=3)
        # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=3)
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics=5, id2word=dictionary)
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
                # print(self.function_id_to_name[f])
                if self.function_id_to_name[f] in function_name_to_docstring:

                    if function_name_to_docstring[self.function_id_to_name[f]] is not None:
                        text_for_summary += function_name_to_docstring[self.function_id_to_name[f]] + ' '
                        # count += 1

        # print(len(text_for_summary))
        print('Cluster comments: ', text_for_summary)

        try:
            cluster_summary = summarize(text_for_summary, ratio=0.35, split=True)
            cluster_summary = ' '.join(list(set(cluster_summary)))
            print('Cluster summary: ', cluster_summary)
            return cluster_summary
        except ValueError:
            return 'Empty'


    def mining_sequential_patterns(self, execution_paths_of_a_cluster):
        """ This function mines sequential patterns from execution paths """
        
        preprocess = [self.execution_paths[item] for item in execution_paths_of_a_cluster]
        
        ps = PrefixSpan(preprocess)

        # ps.maxlen = 15
        ps.minlen = 6

        top5 = ps.topk(10, closed = True)

        # top5 = ps.frequent(2, closed = True)
        
        sentence = ' '
        for i in top5:
            sentence += ' &#187; '
            for j in i[1]:
                sentence += self.function_id_to_name[j] + '(' + self.function_id_to_file_name[j] + ')'
                if j != i[1][len(i[1])-1]:
                    sentence += ' &rarr; '
                
            sentence += ' . <br>'

        return sentence

    def mining_sequential_patterns_from_initial_execution_paths(self, execution_paths):
        ''' This function takes input inital execution paths and outputs frequent mined patterns for 
            focusing the further analysis on important parts
        '''
        number_of_patterns_to_pick = 3000
        extracted_patterns = []
        preprocess = execution_paths
        
        ps = PrefixSpan(preprocess)

        ps.minlen = 20

        ps.maxlen = 30

        top = ps.topk(number_of_patterns_to_pick, closed = True, generator = True)

        for i in top:
            extracted_patterns.append(i[1])

        # print(extracted_patterns)
        return extracted_patterns

    def count_files_in_node(self, execution_paths_of_a_cluster):
        
        files_count = {}
    
        for c in execution_paths_of_a_cluster:
            for f in self.execution_paths[c]:
                if self.function_id_to_file_name[f] in files_count:
                    files_count[self.function_id_to_file_name[f]] += 1
                    
                else:
                    files_count[self.function_id_to_file_name[f]] = 1
                    

        return len(list(files_count.keys())), list(files_count.keys())
        
