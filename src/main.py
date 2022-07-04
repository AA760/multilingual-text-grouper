#%%
import pandas as pd
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer
import keras
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan
import umap
from tqdm import trange
import random

data = pd.read_csv('./data/extracted_n26.csv')
data.rename(columns={data.columns[0]:'FAQ_id'}, inplace=True)


#%%
model = keras.models.load_model("./model")
def embed(input):
    return model(input)

messages = data.title
message_embeddings = embed(messages)


#%%
def generate_clusters(embeddings,
            n_neighbors,
            n_components,
            min_cluster_size,
            random_state=None):
    """
    Generates clusters for input after reducing its dimensions.
    """
    reducer = umap.UMAP(n_neighbors,
                        n_components,
                        metric='cosine',
                        random_state=random_state)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size,
                                metric='euclidean',
                                cluster_selection_method='eom')
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    clusterer.fit(reduced_embeddings)
    
    
    return clusterer


def score_clusters(clusters, prob_threshold = 0.05):
    """
    Returns count of unique clusters and their costs.
    """
    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_samples = len(cluster_labels)
    cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold)/total_samples)
    
    return label_count, cost

def random_search(embeddings, space, num_evals):
    """
    Randomly search hyperparameter space for a limited number of times
    and return a summary of the results.
    """
    results = []
    
    for i in trange(num_evals):
        n_neighbors = random.choice(space['n_neighbors'])
        n_components = random.choice(space['n_components'])
        min_cluster_size = random.choice(space['min_cluster_size'])
        
        clusters = generate_clusters(embeddings,
                                     n_neighbors,
                                     n_components,
                                     min_cluster_size,
                                     random_state=42)
        
        label_count, cost = score_clusters(clusters)
        
        results.append([i, n_neighbors, n_components,
                        min_cluster_size, label_count, cost])
        
    result_df = pd.DataFrame(results, columns=['run_id', 'n_neighbors', 'n_components',
                                                   'min_cluster_size', 'label_count'
                                                   , 'cost'])
        
    return result_df.sort_values(by='cost')


space = {
        "n_neighbors": range(12,16),
        "n_components": range(3,7),
        "min_cluster_size": range(2,16)
        }
search_results = random_search(message_embeddings, space, 100)


#%%
optimal_clusters = generate_clusters(message_embeddings,
                   search_results['n_neighbors'].values[0].item(),
                   search_results['n_components'].values[0].item(),
                   search_results['min_cluster_size'].values[0].item(),
                   random_state=42)

data['FAQ_id'] = optimal_clusters.labels_
