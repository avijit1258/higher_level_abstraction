from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.spatial.distance import pdist

# corpus = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?',]
#
#
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(corpus)
# print(vectorizer.get_feature_names())
# print(X.shape)
# print(vectorizer.vocabulary_)
# print(vectorizer.idf_)

execution_path = [['a', 'b', 'c'], ['d', 'b', 'f'], ['k', 'n', 'l']]

print(execution_path)

#dm = pdist(execution_path,'jaccard')

#print(dm)

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


print(jaccard_distance_matrix(execution_path))