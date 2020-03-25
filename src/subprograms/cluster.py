import re
import os
import sys
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn import preprocessing 
import pandas as pd
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

class DocCluster():
    """
    Class object for text processing, feature vectorization and clustering.
    """
    def __init__(
        self,
        language,
        max_cluster_size
    ):
        self.language = language
        self.max_cluster_size = max_cluster_size
        self.dir = os.path.join(os.getcwd(), self.language)
        self._init_text()

    def _init_text(self):
        self.file_names, self.corpus = self.read_txt(self.dir)

    def read_txt(self, curr_dir):
        """
        Converts local .txt extension files to list of strings

        Parameters: 
        curr_dir (str): Directory to reads .txt files
        Returns: 
        file_names: a list of strings of the file names of .txt files.
        corpus: a list of strings of the text in each file.
        """
        corpus=[]
        file_names=[]
        for filename in os.listdir(curr_dir):
            if filename.endswith(".txt"):
                file_dir = os.path.join(curr_dir, filename)
                with open(file_dir, encoding="utf8") as f:
                    text=f.read()
                    text=re.sub(r'\d+', '', text)
                    corpus.append(text)
                    file_names.append(filename)
            else:
                continue
        file_names = np.array(file_names)

        self.file_names = file_names
        self.corpus = corpus

        return file_names, corpus

    def get_file_names(self):
        return self.file_names

    def get_corpus(self):
        return self.corpus

    def get_feature_vector(self,method):
        """
        Converts the object corpus attribute, a list of strings using the specified feature vectorization method

        Parameters: 
        method (str): Method for embedding. "tf-idf" or "sent_bert" are available at the moment
        Returns: 
        X_Norm: A numpy type array vector with the embeddings.
        """
        print(f"Extracting features from the text data using {method}...")
        if method == "tf-idf":
            self.vectorizer = TfidfVectorizer(
            tokenizer=self.text_preprocessing,
            use_idf=True
            )
            X = self.vectorizer.fit_transform(self.corpus)
            self.X_Norm = preprocessing.normalize(X)
        elif method == "sent_bert":
            model = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')
            self.X_Norm = model.encode(self.corpus)
        else:
            raise TypeError('No method indicated. Please indicate tf-idf or sent_bert to perform feature vectorization.')
        return self.X_Norm

    def text_preprocessing(self, str_input):
        """
        Text Pre-processing a string input for lemmatization, lowercase, stemming, stopwords.

        Parameters: 
        str_input (str): Input must be a string, not a list of strings. String is assumed to be a mixture of words
        Returns: 
        Words (str): Cleaned string data 
        """
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer() 
        blob = TextBlob(str_input.lower())
        tokens = blob.words
        words = [token.stem() for token in tokens]
        words= [lemmatizer.lemmatize(token) for token in tokens]
        words = [w for w in words if not w in stop_words] 
        return words

    def find_optimal_cluster(self, arr):
        """
        Finds the optimal cluster size for K-Means clustering for the input numpy array

        Parameters: 
        arr: Feature Vector Array calculated from corpus
        Returns: 
        optimal_cluster_size: Range of 2 to specified self.max_cluster_size 
        """
        "Calculating silhouette scores for automatic finding of optimal cluster size..."
        optimal=[-1,'']
        for cluster_size in range(2,self.max_cluster_size):
            km = KMeans(n_clusters=cluster_size,random_state=12345)
            km.fit(arr)
            cluster_labels = km.fit_predict(arr)
            silhouette_avg = silhouette_score(arr, cluster_labels)
            print(f"For n_clusters = {cluster_size},The average silhouette_score is :{silhouette_avg}")
            if silhouette_avg>optimal[0]:
                optimal[0],optimal[1] = silhouette_avg,cluster_size
        print(f'Optimal number of clusters is {optimal[1]} with an average silhouette score of {optimal[0]}')
        return optimal[1]

    def get_clusters(self, arr,num_clusters,file_names):
        """
        Apply K-Means with the specified cluster size to group text documents into unssupervised clusters

        Parameters: 
        arr (str): Feature vector
        num_cluster (int): Optimal number of cluster derived from the find_optimal_cluster method, or user specified number of clusters
        file_names (list): A list of strings that contain the file names so that we can create a dataframe for the user to interpret the results
        Returns: 
        df: A pandas dataframe object that contains 2 columns, the file name and the cluster that the file belongs to
        model_labels: A variable that can be fed in to the visualize_clusters method to visualize how well are clusters formed
        """
        km = KMeans(n_clusters=num_clusters,random_state=12345)
        km.fit(arr)
        d={}
        model_labels = km.labels_
        pred_classes = km.predict(arr)
        for cluster in range(num_clusters):
            for file in file_names[np.where(pred_classes == cluster)]:
                d[file] = cluster

        df=pd.DataFrame(list(d.items()), columns=['File', 'Cluster'])
        return df,model_labels

    def visualize_clusters(self,num_clusters,arr,model_labels):
        """
        A basic cluster visualization method adapted from one of the references in the markdown folder.
        For future improvement, labels can be included in each of the points for the user to diagnose individual file discreprencies
        """
        sys.stdout = open(os.devnull, 'w')
        tfs_reduced = TruncatedSVD(n_components=num_clusters, random_state=0).fit_transform(arr)
        tfs_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(tfs_reduced)
        fig = plt.figure(figsize = (10, 10))
        ax = plt.axes()
        plt.scatter(tfs_embedded[:, 0], tfs_embedded[:, 1], marker = ".", c = model_labels)
        plt.show()