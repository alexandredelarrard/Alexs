# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 22:18:10 2018

@author: User
"""

from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import re
from gensim.test.utils import get_tmpfile
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from fastcluster import linkage
from scipy.spatial.distance import cdist, squareform
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import numpy as np
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer

tokenizer = RegexpTokenizer(r"\w+")
stopword_set = set(stopwords.words("french")) 
lemmetizer = FrenchLefffLemmatizer()

def nlp_clean(text):
    text = re.sub(r'\S*@\S*\s?', '', text.strip(), flags=re.MULTILINE) # remove email
    text = re.sub(r'http\S+', '', text, flags=re.MULTILINE) # remove web addresses
    text = re.sub(r'\(Crédits :.+\)', '', text, flags=re.MULTILINE) # remove credits from start of article
    text = re.sub(r'www.\S+', '', text, flags=re.MULTILINE) # remove web addresses
    text = text.lower().translate({ord(ch): None for ch in '0123456789'}) # lower + suppress numbers
    dlist = tokenizer.tokenize(text) # punctuation
#    dlist = list(set(dlist).difference(stopword_set)) # stopwords
    dlist = [lemmetizer.lemmatize(lemmetizer.lemmatize(word, "v"), "n") for word in dlist] ## stemming
    return dlist


class LabeledLineSentence(object):
    def __init__(self, doc_list, id_list = []):
        self.doc_list = doc_list
        self.id = id_list
        self.aftermode = True
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            if not self.aftermode:
                yield TaggedDocument(nlp_clean(doc), [self.id[idx]])
            else:
                yield nlp_clean(doc)
            
def train_doc2vec(full):
    
    articles = pd.read_csv(r"D:\data\articles_journaux\all_articles.csv")["article"].tolist()
    id_list = range(len(articles))
    
    train_corpus = LabeledLineSentence(articles, id_list)
    
    model = Doc2Vec(size = 500, 
                    alpha = 0.025,
                    min_alpha=0.00025,
                    min_count=7,
                    dm =1, 
                    sample = 1e-4,
                    negative = 10,
                    window  = 15,
                    workers=4)
    
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs = 20)
   
    fname = get_tmpfile(r"C:\Users\User\Documents\Alexs\data\doc2vec\v2\doc2vec_articles_181030")
    model.save(fname)
    
def load_model():
    fname = get_tmpfile(r"C:\Users\User\Documents\Alexs\data\doc2vec\v2\doc2vec_articles_181030")
    model = Doc2Vec.load(fname) 
    return model

def transform_article_to_words(x):
    return nlp_clean(x)

def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z
            
        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))
    
def compute_serial_matrix(dist_mat,method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)
        
        compute_serial_matrix transforms a distance matrix into 
        a sorted distance matrix according to the order implied 
        by the hierarchical tree (dendrogram)
    '''
    
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method,preserve_input=True)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
    
    return seriated_dist, res_order, res_linkage

def k_mean(X, full, k = 100):
    
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)    
    
    resultat = pd.DataFrame()
    resultat["article"] = full["article"]
    resultat["url"] = full["url"]
    resultat["k_means_cluster"] = kmeans.labels_
    
    clustering = SpectralClustering(n_clusters=k, assign_labels="discretize", random_state=0).fit(X)
    resultat["Spectral_cluster"] = clustering.labels_
    
    return resultat
    
def similarity_plot(Y):
        
    fig = plt.figure(figsize=(15,15))
    
    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(Y)
    ax.set_aspect('equal')
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()
    
def similarity_cluster(model, data, c):
    
    articles = data.loc[data["cluster"] == c, "article"].tolist()
    
    articl = []
    for a in articles:
        articl.append(a)
    
    X = []
    for x in articl:
        X.append(model.infer_vector(x))
     
    Y = cdist(X,X, metric= "cosine")
    Y = np.round(Y, 6)
    
    return Y, articles
    

if __name__ == "__main__":
    full = pd.read_csv(r"C:\Users\User\Documents\Alexs\data\continuous_run\article\extraction_2018-10-26.csv", sep= "#")
    full = full.loc[~pd.isnull(full["article"])]
    full = full.drop_duplicates("url").reset_index(drop=True)     
    articles = LabeledLineSentence(full["article"].tolist())
    model = load_model()  
   
    Y, reudct = similarity_cluster(model, full, 195)
    
    methods = ["ward","single","average","complete"]
    for method in methods:
        print("Method:\t",method)
        
        ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(Y, method)
        similarity_plot(ordered_dist_mat)
    
    model.docvecs.similarity_unseen_docs(model, articl[275], articl[152])
    resultat.loc[resultat["cluster"] == 12, "article"].tolist()
    
    similarities_0 = []
    for article_i in articl[1:]:
        similarities_0.append(model.docvecs.similarity_unseen_docs(model, articl[0], article_i))
    
    import hdbscan
    
    X = []
    for x in articles:
        X.append(model.infer_vector(x))
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    cluster_labels = clusterer.fit_predict(X)
    full["hdb_cluster"] = cluster_labels
    full["hdb_cluster"].value_counts()    
    
    