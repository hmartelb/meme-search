import os
import pickle
import pandas as pd
import numpy as np

class TopResults():
    def __init__(self, n=1):
        self.top = []
        self.n = n
    
    def insert(self, elements, key):
        if type(elements) != list:
            elements = [elements]
        
        for el in elements:
            if len(self.top) < self.n or el[key] < self.last[key]:
                self.top.append(el)
                self.top = sorted(self.top, key=lambda x: x[key])    
                while len(self.top) > self.n:
                    self.top.pop()
    @property
    def last(self):
        return self.top[-1] if len(self.top) else None

    @property
    def first(self):
        return self.top[0] if len(self.top) else None

    def __repr__(self):
        return str(self.top)

class SearchIndex():
    def __init__(self, features_file=None, index_file=None):
        self.features_file = features_file
        self.index_file = index_file

        self._features = None
        self._index = None

        if self.features_file is not None:
            self.load_features(self.features_file)
        if self.index_file is not None:
            self.load(self.index_file)

    def load_features(self, features_file, column):        
        self._features = pd.read_csv(features_file)
        self._features = self._features[column]
        self._features = self._features.to_numpy()    # TODO: check this!
        
        self.set_features(self._features)

    def set_features(self, features):
        self._features = [{'idx': i, 'data': f} for i,f in enumerate(features)]
        self._feature_dim = len(self._features[0]['data'])

    def load(self, index_file):
        with open(index_file, 'rb') as f:
            self._index = pickle.load(f)
    
    def save(self, index_file):
        assert self._index is not None, 'Index is empty. Call build() before saving'
        with open(index_file, 'wb') as f:
            pickle.dump(self._index, f, protocol=pickle.HIGHEST_PROTOCOL)

    def build(self):
        def _build_kdtree(vectors, depth=0, k=2):
            if not vectors:
                return None
            axis = depth % k
            vectors = sorted(vectors, key=lambda x: x['data'][axis])
            median = len(vectors)//2
            return {
                'point': vectors[median],
                'left':_build_kdtree(vectors[:median], depth + 1, k),
                'right':_build_kdtree(vectors[median + 1:], depth + 1, k)
            }

        self._index = _build_kdtree(self._features, k=self._feature_dim)

    def query(self, vector, n=20, method='efficient'):
        query_fn = self.efficient_search if method == 'efficient' else self.exhaustive_search
        return query_fn(vector, n)

    def efficient_search(self, vector, n=20): 
        '''
        https://gopalcdas.com/2017/05/24/construction-of-k-d-tree-and-using-it-for-nearest-neighbour-search/
        '''
        def _branching_decision(ref, a, b):
            if a is None:
                return b
            if b is None:
                return a
            
            da = self.distance(ref, a['data'])
            db = self.distance(ref, b['data'])

            if da < db:
                return a
            return b

        def _kdtree_closest(results, vector, root, depth=0, k=2):
            if root is None:
                return None
            
            axis = depth % k
            # Calculate branching condition
            if vector[axis] < root['point']['data'][axis]:
                next_branch = root['left']
                opposite_branch = root['right']
            else:
                next_branch = root['right']
                opposite_branch = root['left']

            # Try to insert the current point in the top results
            root['point']['distance'] = self.distance(vector, root['point']['data'])
            results.insert(root['point'], key='distance')

            best = _branching_decision(
                ref=vector,
                a=_kdtree_closest(results, vector, next_branch, depth=depth+1, k=k),
                b=root['point']
            )
            if self.distance(vector, results.last['data']) > (vector[axis] - root['point']['data'][axis]):
                best = _branching_decision(
                    ref=vector, 
                    a=_kdtree_closest(results, vector, opposite_branch, depth=depth+1, k=k), 
                    b=results.last
                )
            return best

        results = TopResults(n)
        _kdtree_closest(results, vector, self._index, k=self._feature_dim)
        return results

    def exhaustive_search(self, vector, n=20):
        results = TopResults(n)
        for feature in self._features:
            feature['distance'] = self.distance(vector, feature['data'])
            results.insert(feature, key='distance')
        return results

    def distance(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        return np.linalg.norm(a-b)

if __name__ == '__main__':
    import time

    # point_list = [(7, 2), (5, 4), (9, 6), (4, 7), (8, 1), (2, 3), (2,2)]
    # point_list = [list(p) for p in point_list]
    # print(point_list)
    
    # si = SearchIndex()
    # si.set_features(point_list)
    # si.build()

    index_name = os.path.join('api', 'images', 'index_4.df')
    df = pd.read_pickle(index_name)['ocr_glove'].to_numpy()

    print("Building index...")
    start = time.time()
    si = SearchIndex()
    si.set_features(df)
    si.build()
    print(f"Build index took {np.round(time.time()-start, 4)} s")

    N = 8
    vec = df[0] # [1,1]# df[0]

    print("Performing queries...")
    start = time.time()
    res_ex = si.query(vec, n=N, method='exhaustive')
    idx_ex = [r['idx'] for r in res_ex.top]
    print('\n', idx_ex)
    print(f"Exhaustive search took {np.round(time.time()-start, 4)} s")

    start = time.time()
    res_ef = si.query(vec, n=N, method='efficient')
    idx_ef = [r['idx'] for r in res_ef.top]
    print('\n', idx_ef)
    print(f"Efficient search took {np.round(time.time()-start, 4)} s")

    assert idx_ex == idx_ef, 'Search retured different results!'