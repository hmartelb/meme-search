import logging
import os
import pickle
import string
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from scipy.spatial import distance

# import scipy
from config import *
from features import SentenceVectorizer
from scipy_search import SearchIndex

app = FastAPI()
logger = logging.getLogger(__name__)

varables = {}
sv = SentenceVectorizer()
search_index = SearchIndex()

@app.get('/initialize')
@app.on_event("startup")
def initialize():
    reload_index()
    reload_sentence_vectorizer()
    
@app.get('/reload_sentence_vectorizer')
def reload_sentence_vectorizer():
    global sv
    try:
        sv.load(EMBEDDINGS_FILENAME)
    except:
        logger = logging.getLogger('uvicorn.error')
        logger.error('Failed to load vectors')
        return 'Failed to load vectors'

    logger = logging.getLogger('uvicorn.info')
    logger.info(f'Success loading vectors: {EMBEDDINGS_FILENAME}')
    return 'Success loading vectors'

@app.get('/reload_index')
def reload_index():
    global search_index
    try:
        search_index.load(filename=SEARCH_INDEX_FILENAME, reader_fn=SEARCH_READER_FN)
        search_index.build(search_cols=SEARCH_COLUMNS ,max_dim=SEARCH_MAX_DIM)
    except:
        logger = logging.getLogger('uvicorn.error')
        logger.error('Failed to load index')
        return 'Failed to load index'

    logger = logging.getLogger('uvicorn.info')
    logger.info(f'Success loading index: {SEARCH_INDEX_FILENAME}')
    return 'Success loading index'

@app.get('/')
def index(query: str, count: int = 20, mode: str = 'both', threshold: float = 1.0):

    if mode == 'both': search_column = 'fusion_text_glove'
    if mode == 'title': search_column = 'title_glove'
    if mode == 'content': search_column = 'ocr_glove'

    if query:
        start = time.time()
        
        # Calculate the embedding of the query
        query_embedding = sv.encode(query)

        # Perform the query in the index
        query_results, scores = search_index.query(
            vector=query_embedding, 
            col=search_column, 
            k=count, 
            return_scores=True
        )

        # Put results in adequate format
        results = []
        for (i, item), score in zip(query_results.iterrows(), scores):
            if score <= threshold:
                results.append({
                    'name': item['title'],
                    'url': item['media'],
                    'score': score
                })

        logger = logging.getLogger('uvicorn.info')
        logger.info(f'QUERY: "{query}" ({np.round(time.time()-start, 4)} s., top-{count})')
        
        return {'results': results }

    return 'Hello from FastAPI'    

if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=10000)
    ap.add_argument('--gpu', default='-1')
    args = ap.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Launch app
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
