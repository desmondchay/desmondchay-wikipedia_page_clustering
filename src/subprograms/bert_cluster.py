"""
This examples clusters different sentences that come from the same wikipedia article.
It uses the 'wikipedia-sections' model, a model that was trained to differentiate if two sentences from the
same article come from the same section or from different sections in that article.
"""

from sentence_transformers import SentenceTransformer
# from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import re
import os
import string

embedder = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')

os.chdir(r"C:\Users\desmond\Desktop\dathena\wiki corpus\en")
curr_dir=os.getcwd()
corpus = []

for filename in os.listdir(curr_dir):
    if filename.endswith(".txt"): 
        with open(filename,encoding="utf8") as f:
            text=f.read()
            text = text.translate(str.maketrans('', '', string.punctuation))
            text=re.sub('\s+',' ',text)
            text=text.strip().lower()
            corpus.append(text)
    else:
        continue
corpus[0]
corpus_embeddings = embedder.encode(corpus)
len(corpus_embeddings[0])
# Perform kmean clustering
num_clusters = 5
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i+1)
    print(cluster)
    print("")
