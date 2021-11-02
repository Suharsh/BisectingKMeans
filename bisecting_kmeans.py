import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

with open("train.dat", "r", encoding="utf8") as file:
    lines = file.readlines()

docs = [line.split() for line in lines]

def convert_to_csr_matrix(docs):
    rows = [] #doc number
    cols = [] #term indexes
    data = [] # frquency values
    for docno, doc in enumerate(docs):
        for idx in range(int(len(doc)/2)):
            rows.append(docno)
            cols.append(int(doc[idx*2])-1)
            data.append(float(doc[idx*2+1]))
    print(len(rows),len(cols),len(data))
    csr_mat = csr_matrix((data, (rows, cols)), dtype = np.single)
    return csr_mat

def convert_csr_matrix_2_np_array(csr_matrix):
    npa = np.array(csr_matrix)
    return npa

docs_csr = convert_to_csr_matrix(docs)
docs_csr.toarray().shape

docs_npa = convert_csr_matrix_2_np_array(docs_csr)

transformer = TfidfTransformer() # go with default values norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False
docs_tfidf = transformer.fit_transform(docs_csr)

docs_tfidf_arr = docs_tfidf.toarray()

svd = TruncatedSVD(n_components=2200)
docs_svd = svd.fit_transform(docs_tfidf_arr)
var_explained = svd.explained_variance_ratio_.sum()
print(var_explained)

def getActualIndices(cluster_index, labels):
    return np.where(labels == cluster_index)[0]

def bisecting_kmeans(k,data,n_iter):
    clusters = []
    selected_cluster_indices = []
    for i in range(data.shape[0]):
        selected_cluster_indices.append(i)

    while len(clusters) < k:
        selected_cluster_data = data[selected_cluster_indices,:]
        kmeans = KMeans(n_clusters=2,n_init=n_iter,random_state=42).fit(selected_cluster_data)
        kmeans_cluster_centre = kmeans.cluster_centers_
        SSE = [0,0];
        for point,label in zip(data,kmeans.labels_):
            SSE[label]+=np.square(point-kmeans_cluster_centre[label]).sum()
        selected_cluster_index = np.argmax(SSE,axis=0)
        dropped_cluster_index = 1 if selected_cluster_index == 0 else 0
        selected_cluster = getActualIndices(selected_cluster_index,kmeans.labels_)
        dropped_cluster = getActualIndices(dropped_cluster_index,kmeans.labels_)
        actual_selected_cluster = []
        actual_dropped_cluster = []
        for index in selected_cluster:
            actual_selected_cluster.append(selected_cluster_indices[index])

        for index in dropped_cluster:
            actual_dropped_cluster.append(selected_cluster_indices[index])
        clusters.append(actual_dropped_cluster)
        selected_cluster_indices = actual_selected_cluster

    labels = [0] * data.shape[0]
    for index, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = index + 1
    return labels

k_values = list()
s_scores = list()
for k in range(3, 22, 2):
    labels = bisecting_kmeans(k, docs_svd,10)
    if (k == 11):
        outputFile = open("kmeans_suharsh_test_11_2000.dat", "w")
        for index in labels:
            outputFile.write(str(index) +'\n')
        outputFile.close()
    score = silhouette_score(docs_svd, labels,metric='euclidean')
    k_values.append(k)
    s_scores.append(score)
    print ("Silhouette_coefficient is %f for K= %d  " %(score,k))


