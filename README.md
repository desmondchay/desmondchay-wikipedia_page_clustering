---
## Automatic Document Clustering
---
## Setup
1. Navigate to root folder
2. Install your preferred torch version on https://pytorch.org/get-started/locally/
3. pip install -r requirements.txt
4. Run the following.
```
# run program
cd src
python main.py
```

---
### Scripts
- scraper.py
- language_detection.py
- cluster.py
---
### Document Ingestion
Extract Wikipedia page summary and convert it to .txt files. Search terms, number of results & languages that can be altered through config.py

The function is disabled by default. Uncomment scraper.py in the main.py method to scrap for additional articles.

Data is saved into the files subfolder in the src folder by default

The wikipedia package was used to facilitate scraping of wikipedia pages,
due to its ease of use and the nature of data obtained using the API, which reduces the need for data cleaning.

- Script Name: scraper.py
- Source: Wikipedia
- Third-party packages used: wikipedia

---
### Language Detection
Sort .txt files by document average language, implemented using spacy.
Creates a new sub-folder in files for every new language detected.

The spacy library was used again, due to ease of implementation. It also supports multiple languages out of the box.

- Script Name: language_detection.py
- Third-party packages used: spacy, spacy_langdetect
- Language Scope: All

---
### Text Processing
Numerical values are removed for ease of processing, on the assumption that they do not provide much information specifically for document embedding. 

Words were converted to lowercase, then stemmed and lemmatize to reduce the vocabulary size and aid with processing. No tokenizing is done at this stage as an exernal package from sklearn, TfidfVectorizer, is able to accept inputs of strings for conversion to tf-idf vectors. English stopwords were obtained from nltk.corpus.

No removal of stopwords is done for BERT as the algorithm benefits from the context, which will be non-existent if stopwords were to be removed. 

For the specify handling of documents of different languages thereafter, the analysis will focus on the english document as the inspection of the results of the cluster analysis is easier done due to a better english comprehension level of the author.

- Script Name: cluster.py
- Third-party packages used: sklearn, textblob, nltk
---
### Feature Vectorization & Clustering
There are 2 methods of feature vectorization, one using term document-inverse document frequency (td-idf) with another using BERT, specifically a model trained
with wikipedia articles, that is more suited for the context of this task.

K-means clustering was used as the clustering algorithm. It was used based on an assumption of even cluster size, with a relatively low number of clusters intrisically.
Distance metric specified was euclidean, though the normalization of the tf-idf vector generated allow the clustering based on cosine similarity instead.

The DocCluster class, along with its various methods in cluster.py enables the user to configure the method of feature vectorization. It encompasses methods for K-means clustering, feature vectorization with either sent_bert or tf-idf, and a method that uses silhouette scores to find the optimal number of clusters to be fitted for K-means. Silhouette scores were used as it is the most convenient metric for K-means, though it may not be suited for text data because of the high dimensionality which will increase computation times.

- Script Name: cluster.py
- Third-party packages used: sklearn, nltk, pandas, numpy, matplotlib, sentence_transformers, textblob
---
### Results

A total of 265 different english documents were used in the clustering. With the number of cluster given through the find_optimal_cluster method, as required by the K-means clustering algorithm, a range of 4-6 clusters were obtained for tf-idf, while sent_bert generally produces a range of 2-5 clusters.

#### __tf-idf results__

|  File                          |   Cluster |
| :------------------------------|----------:|
|  Act of God.txt                |         0 |
|  Breach of contract.txt        |         0 |
|  Conflict of contract laws.txt |         0 |
|  Contract (canon law).txt      |         0 |
|  Contract Clause.txt           |         0 |

|  File                                          |   Cluster |
| :----------------------------------------------|----------:|
|  Alienation (property law).txt                 |         1 |
|  Ancient Norwegian property laws.txt           |         1 |
|  Association for Law, Property and Society.txt |         1 |
|  Australian property law.txt                   |         1 |
|  Canadian intellectual property law.txt        |         1 |

|  File                                      |   Cluster |
| :------------------------------------------|----------:|
|  Age of majority.txt                       |         2 |
|  Animals, Property, and the Law.txt        |         2 |
|  Application of Islamic law by country.txt |         2 |
|  Bankruptcy.txt                            |         2 |
|  Basque civil law.txt                      |         2 |

|  File                                         |   Cluster |
| :---------------------------------------------|----------:|
|  Aerial bombardment and international law.txt |         3 |
|  Cannabis and international law.txt           |         3 |
|  Centre for International Law.txt             |         3 |
|  Customary international humanitarian law.txt |         3 |
|  Customary international law.txt              |         3 |

|  File                              |   Cluster |
| :----------------------------------|----------:|
|  Bengal Criminal Law Amendment.txt |         4 |
|  California criminal law.txt       |         4 |
|  Consent (criminal law).txt        |         4 |
|  Conspiracy (criminal).txt         |         4 |
|  Crime.txt                         |         4 |

#### __sent_bert results__

|  File                                |   Cluster |
| :------------------------------------|----------:|
|  Cannabis and international law.txt  |         0 |
|  Civil union.txt                     |         0 |
|  Community property.txt              |         0 |
|  Condominium (international law).txt |         0 |
|  Conflict of laws.txt                |         0 |

|  File                                         |   Cluster |
| :---------------------------------------------|----------:|
|  Aerial bombardment and international law.txt |         1 |
|  Age of majority.txt                          |         1 |
|  Basque civil law.txt                         |         1 |
|  California criminal law.txt                  |         1 |
|  Canadian intellectual property law.txt       |         1 |

|  File                                      |   Cluster |
| :------------------------------------------|----------:|
|  Ancient Norwegian property laws.txt       |         2 |
|  Application of Islamic law by country.txt |         2 |
|  Bengal Criminal Law Amendment.txt         |         2 |
|  Canadian contract law.txt                 |         2 |
|  Chinese property law.txt                  |         2 |

|  File                    |   Cluster |
| :------------------------|----------:|
|  Act of God.txt          |         3 |
|  Bankruptcy.txt          |         3 |
|  Breach of contract.txt  |         3 |
|  Coercion.txt            |         3 |
|  Common-law marriage.txt |         3 |

|  File                                          |   Cluster |
| :----------------------------------------------|----------:|
|  Animals, Property, and the Law.txt            |         4 |
|  Association for Law, Property and Society.txt |         4 |
|  Centre for International Law.txt              |         4 |
|  Civil Law Commentaries.txt                    |         4 |
|  Civil Law Initiative.txt                      |         4 |

|  File                                 |   Cluster |
| :-------------------------------------|----------:|
|  Alienation (property law).txt        |         5 |
|  Australian property law.txt          |         5 |
|  Canadian property law.txt            |         5 |
|  Civil engineering.txt                |         5 |
|  Copyright law of the Philippines.txt |         5 |

We can briefly summarize the nature of each cluster by inspection, based on the sample of documents in each cluster. The clusters can be separated based on the gist of the content given by their file name. Clusters can be termed as property law cluster, civil law cluster, international law cluster, cluster contract law cluster and lastly, criminal law cluster. 

By manual inspection, tf-idf seem to give the most promising results, as it produces clusters with majority of its documents related to each other. Cluster 0 for tf-idf refers to contract law documents, Cluster 1 on property law, Cluster 2 contains a mixture of documents that do not primarily fall into any main group, Cluster 3 on international law and finally Cluster 4 on criminal law. A brief examination of the first few clusters in each cluster presents that clusters are well formed.

For documents where BERT feature vectorization is applied, the clusters formed seem very different from the clusters formed by tf-idf. Each cluster has a mixture of different documents belonging to general law specialy, other than cluster 5, where the documents seem to be generally of the property law nature. This may be because the BERT feature vectorization used is pre-trained on wikipedia sections instead of whole documents, which may affect the difference in subtleties of semantics of one over the other. Furthermore, the BERT model has not been trained on such documents, which may be the reason why by visual inspection, tf-idf seem to perform better in this clustering task. 

While the results may be promising, it is advisable to refine the analysis through a comprehensive literature review of the effectiveness of different document embedding techniques, like word2vec, or even topic modelling algorithms like LDA. Dimensionality reduction can also be achieved using LSA.

Other clustering algorithms like hierachical clustering, DBSCAN can be attempted as well, though the effectiveness of different clustering algorithms will be dependent on the distribution of type of documents, as well as the number of documents within the corpus. 

---
### References
- https://wikipedia.readthedocs.io/en/latest/code.html
- https://spacy.io/universe/project/spacy-langdetect
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- https://github.com/UKPLab/sentence-transformers
- https://beckernick.github.io/law-clustering/