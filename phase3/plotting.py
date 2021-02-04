from yellowbrick.cluster import kelbow_visualizer, intercluster_distance
from sklearn.cluster import KMeans

from .preprocess import *
from .clustering import return_clustered_csv, kmeans, load_data, vectorize, get_res

data, major_labels, minor_labels = load_data(stem=False, lemmatize=True, remove_conjunctions=True)
tfidf, w2v = vectorize(data, w2v_options=dict(size=64, iter=128, min_count=2))
pca_tfidf = pca(64, tfidf.toarray(), random_state=666)

get_res(return_clustered_csv(data, kmeans, pca_tfidf, w2v, options=dict(random_state=666), save=False), data=data)

print("TF-IDF:\n")
_ = kelbow_visualizer(KMeans(random_state=666), pca_tfidf, k=(2, 16), metric='silhouette')
_ = intercluster_distance(KMeans(n_clusters=5, random_state=666), pca_tfidf)
print("\nW2V:\n")
_ = kelbow_visualizer(KMeans(random_state=666), w2v, k=(2, 16), metric='silhouette')
_ = intercluster_distance(KMeans(n_clusters=5, random_state=666), w2v)

