import re
import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn import preprocessing 
import numpy as np

def read_txt(curr_dir):
    """TBF"""
    corpus=[]
    file_names=[]
    for filename in os.listdir(curr_dir): 
        if filename.endswith(".txt"): 
            with open(filename,encoding="utf8") as f:
                text=f.read()
                text=re.sub(r'\d+', '', text)
                corpus.append(text)
                file_names.append(filename)
        else:
            continue
    file_names=np.array(file_names)
    return file_names,corpus

def textblob_tokenizer(str_input):
    """TBF"""
    lemmatizer=WordNetLemmatizer() 
    blob=TextBlob(str_input.lower())
    tokens=blob.words
    words=[token.stem() for token in tokens]
    words=[lemmatizer.lemmatize(token) for token in tokens]
    return words

def tfidf_vectorize(corpus):
    """TBF"""
    vectorizer = TfidfVectorizer(tokenizer=textblob_tokenizer,stop_words='english',use_idf=True,)
    X = vectorizer.fit_transform(corpus)
    X_Norm = preprocessing.normalize(X)
    return X_Norm,vectorizer

#Finding optimal cluster size
def find_optimal_cluster(arr,max_clusters):
    """TBF"""
    distorsions=[]
    for k in range(2, 20):
        km = KMeans(n_clusters=max_clusters)
        km.fit(arr)
        distorsions.append(km.inertia_)
    return distorsions

def get_clusters(num_clusters,arr,vectorizer,file_names):
    """TBF"""
    km = KMeans(n_clusters=num_clusters)
    km.fit(arr)
    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(num_clusters):
        top_ten_words = [terms[ind] for ind in order_centroids[i, :10]]
        print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))
        
    d={}
    pred_classes=km.predict(arr)
    for cluster in range(num_clusters):
        for file in file_names[np.where(pred_classes == cluster)]:
            d[file]=cluster

    results=pd.DataFrame(list(d.items()), columns=['File', 'Cluster'])
    return results
