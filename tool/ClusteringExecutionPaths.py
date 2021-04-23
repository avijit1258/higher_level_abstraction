from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, to_tree
from scipy.spatial import distance as ssd

class ClusteringExecutionPath:

    def __init__(self):
        self.cluster_to_leaf = {}
        self.leaf_to_cluster = {}

    def flat_cluster(self, mt):

        Z = linkage(ssd.squareform(mt), 'ward')

        flat_cluster = fcluster(Z, t= 5, criterion='distance')
        # flat_cluster = fcluster(Z, t = 4, monocrit= 'maxclust')

        for leaf, cluster in enumerate(flat_cluster):
            if cluster in self.cluster_to_leaf:
                self.cluster_to_leaf[cluster].append(leaf)
            else:
                self.cluster_to_leaf[cluster] = [leaf]
            
            self.leaf_to_cluster[leaf] = cluster

        return flat_cluster

    def label_flat_clusters(self, document_nodes, mat):
        tree = []

        self.flat_cluster(mat)

        for cluster, leaves in self.cluster_to_leaf.items():
            tree.append( document_nodes.labeling_cluster(leaves, cluster, -1))

        return tree


        