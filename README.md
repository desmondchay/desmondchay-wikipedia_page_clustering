---
## Dathena Automatic Document Clustering
---
---
## Setup
1. Navigate to root folder
2. pip install -r requirements.txt
3. Run main.py

---
### Scripts
- scrapper.py
- language_detection.py
- tfidf_sklearn.py
---
### Document Ingestion
Extract Wikipedia page summary and convert it to .txt files. Search terms, number of results & languages that can be altered through config.py

- Script Name: scrapper.py
- Source: Wikipedia
- Third-party packages used: wikipedia

---
### Language Detection
Sort .txt files by document average language, implemented using spacy.
Creates a new sub-folder for every new language detected.

- Script Name: language_detection.py
- Third-party packages used: spacy, spacy-langdetect
- Language Scope: All

---
### Text Processing
Numerical values are removed for ease of processing, on the assumption that they do not provide much information specifically for document embedding. 

Words were converted to lowercase, then stemmed and lemmatize to reduce the vocabulary size and aid with processing. No tokenizing is done at this stage as an exernal package from sklearn, TfidfVectorizer, is able to accept inputs of strings for conversion to tf-idf vectors.

- Script Name: language_detection.py
- Third-party packages used: spacy, spacy-langdetect
- Language Scope: All
---
### Text Processing
Numerical values are removed for ease of processing, on the assumption that they do not provide much information specifically for document embedding. 

Words were converted to lowercase, then stemmed and lemmatize to reduce the vocabulary size and aid with processing. No tokenizing is done at this stage as an exernal package from sklearn, TfidfVectorizer, is able to accept inputs of strings for conversion to tf-idf vectors.

For the specify handling of documents of different languages thereafter, the analysis will focus on the english document as the inspection of the results of the cluster analysis is easier done due to a better english comprehension level of the author.

- Script Name: tfidf_sklearn.py
- Third-party packages used: , spacy-langdetect
---
### Feature Vectorization & Clustering
tf-idf was computed for the vocabulary in the text corpus to be used as the input vector for clustering. 

-INSERT WHY WAS TF-IDF USED-

K-means clustering was used as the clustering algorithm. It was used based on an assumption of even cluster size, with a relatively low number of clusters intrisically.

Distance metric specified was euclidean, though the normalization of the tf-idf vector generated allow the clustering based on cosine similarity instead.

- Script Name: tfidf_sklearn.py
- Third-party packages used: sklearn, nltk, pandas, numpy
---
### Results

A total of 313 different english documents were used in the clustering. With the number of cluster specified to be 4 as required by the K-means clustering algorithm, the following clusters were obtained, with a sample of documents that are clustered in each cluster.

| Cluster 0 (45) | Cluster 1 (128)| Cluster 2 (42) | Cluster 3 (52)| Cluster 4 (46) |
| ------ | ----------- |------ | ----------- | ----------- |
|Accession (property law).txt|A-law algorithm.txt|Aerial bombardment and international law.txt|Aleatory contract.txt|Automatism (law).txt|
|Alienation (property law).txt|Abortion law.txt|Cannabis and international law.txt|Alien Contract Labor Law.txt|California criminal law.txt|
|Ancient Norwegian property laws.txt|Actor in Law.txt|Centre for International Law.txt|Australian contract law.txt|Consent (criminal law).txt|
|Association for Law, Property and Society.txt|Administrative law.txt|Conflict of laws.txt|Breach of contract.txt|Conspiracy (criminal).txt|
|Australian property law.txt|Animals, Property, and the Law.txt|Constitutional law.txt|Canadian contract law.txt|Crime.txt|

We can briefly summarize the nature of each cluster based on the sample of documents in each cluster. Cluster 0 is mainly documents on property law, cluster 1 on civil law, cluster 2 on international law, cluster 3 on contract law ad lastly, cluster 4 is on criminal law.

While the results may be promising, it is advisable to refine the analysis through a comprehensive literature review of the effectiveness of different document embedding techniques, like doc2vec, BERT, or even topic modelling algorithms like LDA. 

Other clustering algorithms like hierachical clustering can be attempted as well, though the effectiveness of different clustering algorithms will be dependent on the distribution of type of documents within the corpus. 

---
### References
- https://wikipedia.readthedocs.io/en/latest/code.html
- https://spacy.io/universe/project/spacy-langdetect
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html