from yellowbrick.cluster import kelbow_visualizer, intercluster_distance
from sklearn.cluster import KMeans

from .preprocess import *
from .clustering import return_clustered_csv, kmeans

print("TF-IDF:\n")
_ = kelbow_visualizer(KMeans(random_state=666), pca_tfidf, k=(2, 16), metric='silhouette')
_ = intercluster_distance(KMeans(n_clusters=5, random_state=666), pca_tfidf)
print("\nW2V:\n")
_ = kelbow_visualizer(KMeans(random_state=666), w2v, k=(2, 16), metric='silhouette')
_ = intercluster_distance(KMeans(n_clusters=5, random_state=666), w2v)

