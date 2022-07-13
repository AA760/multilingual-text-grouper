#%%
import pandas as pd
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer
import keras
import hdbscan
import umap
from sklearn.model_selection import ParameterGrid

data = pd.read_csv('./data/extracted_n26.csv')
data.rename(columns={data.columns[0]:'FAQ_id'}, inplace=True)


#%%
try: 
    model = keras.models.load_model("./model")
except:
    print("Local model not found, downloading instead...")
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    
def embed(input):
    return model(input)

messages = data.title
message_embeddings = embed(messages)


#%%

def umap_hdb(n_neighbors, n_components):
    """
    Generates clusters for input after reducing its dimensions.
    """
    reducer = umap.UMAP(n_neighbors,
                        n_components,
                        metric='cosine',
                        random_state=None)

    """
    min_cluster_size has to be at least 2;
    outliers (labelled as -1) will have to be dealt with after.
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2,
                                metric='euclidean',
                                cluster_selection_method='eom')
    
    reduced_embeddings = reducer.fit_transform(message_embeddings)
    clusterer.fit(reduced_embeddings)
    
    return clusterer


def score_clusters(clusterer, prob_threshold = 0.05):
    """
    Returns count of unique clusters and their costs.
    """
    cluster_labels = clusterer.labels_
    label_count = len(np.unique(cluster_labels))
    total_samples = len(cluster_labels)
    cost = (np.count_nonzero(clusterer.probabilities_ < prob_threshold)/total_samples)
    
    return label_count, cost

param_grid = {'n_neighbors':range(2,100),
              'n_components':range(2,10)}

param_grid = ParameterGrid(param_grid)

results = []
i = 0


for params in param_grid:
    label_count, cost = score_clusters(umap_hdb(**params))
    results.append([i, param_grid[i], label_count, cost])
    i = i + 1

#%%
result_df = pd.DataFrame(results, columns=['run_id', 'params', 'label_count',
                                               'cost'])
result_df = result_df.sort_values(by='cost')
print(result_df)


#%%
best_params = result_df.iloc[0,1]
print(best_params)
best_clusters = umap_hdb(best_params["n_neighbors"], best_params["n_components"])
data.FAQ_id = best_clusters.labels_

