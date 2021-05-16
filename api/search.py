import os

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.decomposition import PCA


class SearchIndex():
    """
    Perform quick searches over high-dimensional vector fields of a pd.DataFrame 
    using scipy.spatial.KDTree as indexing. 

    This class provides methods to load the data into a pd.DataFrame and build the 
    index for the columns specified by the user. After building the index, it can
    be used to perform k-nearest neigbors queries by column.

    Parameters
    ----------
    filename (str):         Location of the file to be read into the pd.DataFrame.
    reader_fn (callable):   Function to parse the file (default: pd.read_csv).
    search_cols (list):     Columns to use for the searches. Must be present in the 
                            resulting dataframe.
    max_dim (int):          Maximum number of dimensions of the indexed data. If the 
                            data has higher dimensions, PCA will be applied to transform
                            it into max_dim (default: 300).
    """

    def __init__(self, filename=None, search_cols=[], max_dims=[], reader_fn=pd.read_csv):
        # assert filename is not None and search_cols == [], 'You must provide search columns'

        self.filename = filename
        self.search_cols = search_cols

        self.data = None
        self.trees = {}
        self.pcas = {}

        if len(max_dims) == len(search_cols):
            self.max_dims = {sc:dim for sc,dim in zip(search_cols, max_dims)} 
        else:
            self.max_dims = {sc:300 for sc in search_cols}

        if filename is not None:
            self.load(filename, reader_fn=reader_fn)

            if self.search_cols != []:
                self.build()

    def build(self, search_cols=[], max_dims=[]):
        if search_cols != []:
            self.search_cols = search_cols
        assert len(self.search_cols) > 0, 'Empty columns, cannot build index'

        if max_dims is not None and len(search_cols) == len(max_dims):
            self.max_dims = {sc:dim for sc,dim in zip(search_cols, max_dims)} 

        for col in self.search_cols:
            # Convert the values of the dataframe in a 2D numpy array shape=(samples, dim)
            features = np.stack([np.array(item) for item in self.data[col]])
            features = np.nan_to_num(features)

            # Apply dimensionality reduction (PCA) if the dimensionality of the data is too high (dim > self.max_dims)
            if features.shape[1] > self.max_dims[col]:
                pca = PCA(n_components=self.max_dims[col])
                self.pcas[col] = pca.fit(features)
                features = self.pcas[col].transform(features)

            # Build the tree and save it
            self.trees[col] = KDTree(features)

    def load(self, filename, reader_fn=pd.read_csv):
        self.data = reader_fn(filename)
        # self.data = self.data.reset_index()

    def query(self, vector, col, k=20, return_scores=False, threshold=None):
        assert col in self.trees.keys(), f'Wrong column, {col} is not indexed'

        # Apply the same transform to the query vector if the dimensionality is too high (len(vector) > self.max_dims)
        if len(vector) > self.max_dims[col]:
            vector = self.pcas[col].transform(vector.reshape(1, -1)).reshape(-1)

        # Perform the query in the KDTree of the corresponding column
        if threshold is None:
            scores, idx = self.trees[col].query(vector, k=k)
        else:
            scores, idx = self.trees[col].query(vector, k=k, distance_upper_bound=threshold)

        # Retrieve entries from the original data
        results = self.data.iloc[idx]
        if return_scores:
            return results, scores  
        return results


if __name__ == '__main__':
    import time
    from sentence_vectorizer import SentenceVectorizer
   
    print("Building index...")
    start = time.time()

    search_index = SearchIndex(
        filename=os.path.join('api', 'images', 'index_4.df'),
        search_cols=['fusion_text_glove', 'ocr_glove'],
        reader_fn=pd.read_pickle,
        max_dim=300
    )
    search_index.build()

    # tree = KDTree(features, leafsize=100)
    print(f"Build index took {np.round(time.time()-start, 4)} s")

    vectors_filename = os.path.join(
        'api', 'pretrained', 'glove.6B.300d_dict.pickle')
    sv = SentenceVectorizer(filename=vectors_filename)

    query = 'who would win?'
    query_vector = sv.encode(query)

    print("Performing queries...")
    start = time.time()
    results = search_index.query(query_vector, col='fusion_text_glove', k=20)
    print(f"KDTree search took {np.round(time.time()-start, 4)} s")

    print(results.head(20))