import os
os.chdir(r'C:\Users\desmond\Desktop\dchay\src')
ROOT=os.getcwd()
import config as cfg
import subprograms.scrapper as scrapper
import subprograms.language_detection as language_detection
import subprograms.tfidf_sklearn as cluster_model

languages,search_term,num_results=cfg.scrapper['languages'],cfg.scrapper['search_term'],cfg.scrapper['num_results']
os.chdir(os.path.join(ROOT,'data'))

#Modules to get data and sort to subfolders by document language
# scrapper.make_corpus(languages,search_term,num_results) 
language_detection.sort_by_language()


os.chdir(os.path.join(ROOT,'data','en'))
curr_dir=os.getcwd()
file_names,corpus=cluster_model.read_txt(curr_dir)
X_Norm,vectorizer=cluster_model.tfidf_vectorize(corpus)

# To determine optimal number of clusters via visual inspection
# from matplotlib import pyplot as plt
# distorsions=cluster_model.find_optimal_cluster(X_Norm,10)
# fig = plt.figure(figsize=(15, 5))
# plt.plot(range(2, 10), distorsions)
# plt.grid(True)
# plt.title('Elbow curve')

num_clusters=5
results=cluster_model.get_clusters(num_clusters,X_Norm,vectorizer,file_names)

results.count()
results[results['Cluster']==0].head()
results[results['Cluster']==1].head()
results[results['Cluster']==2].head()
results[results['Cluster']==3].head()
results[results['Cluster']==4].head()