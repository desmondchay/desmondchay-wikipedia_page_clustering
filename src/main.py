import os

import subprograms.config as cfg
from subprograms.scraper import scraper
from subprograms.language_detection import language_detector
from subprograms.cluster import DocCluster

def main():
    # Performs scraping. Comment out to disable
    languages, search_term, num_results = cfg.params['languages'], cfg.params['search_term'], cfg.params['num_results']
    # scraper()

    ROOT = os.getcwd()
    os.chdir(os.path.join(ROOT, 'files'))

    # Detect language and sort to subfolders by document language
    language_detector()

    # Initializes class 
    cluster_model = DocCluster(language='en',max_cluster_size=20)

    # Reads in input files to obtain a list of strings, then apply text pre-processing feature vectorization
    feature_vector=cluster_model.get_feature_vector(method='sent_bert')

    # Cluster documents into logical groups by using cluster size with highest average silhouette score as optimal. KMeans is used.
    num_clusters=cluster_model.find_optimal_cluster(feature_vector)
    file_names = cluster_model.get_file_names()
    results_df,model_labels = cluster_model.get_clusters(feature_vector,num_clusters,file_names)

    cluster_model.visualize_clusters(num_clusters,feature_vector,model_labels)
    print("Number of documents analysed:",results_df.count()[1])
    
    for cluster in results_df['Cluster'].unique():
        print(results_df[results_df['Cluster']==cluster].head())

if __name__ == "__main__":
    main()