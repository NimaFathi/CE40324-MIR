import seaborn as sns

from yellowbrick.cluster import kelbow_visualizer, intercluster_distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# from .preprocess import *
from .clustering import return_clustered_csv, kmeans, load_data, vectorize, get_res, grid_search

sns.set_theme()
sns.set_style("dark")

# loading data
data, major_labels, minor_labels = load_data(stem=False, lemmatize=True, remove_conjunctions=True)
tfidf, w2v = vectorize(data, w2v_options=dict(size=64, iter=128, min_count=2))
pca_tfidf = PCA(64, random_state=666).fit_transform(tfidf.toarray())

# cluster diagrams
get_res(return_clustered_csv(data, kmeans, pca_tfidf, w2v, options=dict(random_state=666), save=False), data=data)

# elbow charts and Inter-cluster Distance Maps
print("TF-IDF:\n")
_ = kelbow_visualizer(KMeans(random_state=666), pca_tfidf, k=(2, 16), metric='silhouette')
_ = intercluster_distance(KMeans(n_clusters=5, random_state=666), pca_tfidf)
print("\nW2V:\n")
_ = kelbow_visualizer(KMeans(random_state=666), w2v, k=(2, 16), metric='silhouette')
_ = intercluster_distance(KMeans(n_clusters=5, random_state=666), w2v)

# changing the number of clusters
variables = dict(n_components=list(range(4,16)), max_iter=[100, 300, 600], n_init=[5,10,15,20])
fixed_vars = dict(random_state=666)
res = grid_search(kmeans, data=data, tfidf=pca_tfidf, w2v=w2v, variables=variables,
                  fixed_params=fixed_vars)
_ = sns.relplot(data=res, x='n_components', y='score', row='vectorization', col='max_iter', hue="metric", kind='line')
