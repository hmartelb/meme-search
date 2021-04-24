import logging
import os
import pickle
import string

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from scipy.spatial import distance

# import scipy
from config import *
from features import SentenceVectorizer

app = FastAPI()
logger = logging.getLogger(__name__)

varables = {}
sv = SentenceVectorizer()
search_index = None

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
        search_index = pd.read_pickle(INDEX_FILENAME)
    except:
        logger = logging.getLogger('uvicorn.error')
        logger.error('Failed to load index')
        return 'Failed to load index'

    logger = logging.getLogger('uvicorn.info')
    logger.info(f'Success loading index: {INDEX_FILENAME}')
    return 'Success loading index'

@app.get('/')
def index(query: str, count: int = 20, mode: str = 'both', threshold: float = 1.0):

    if mode == 'both': search_column = 'fusion_text_glove'
    if mode == 'title': search_column = 'title_glove'
    if mode == 'content': search_column = 'ocr_glove'

    if query:
        # Calculate the embedding of the query
        query_embedding = sv.encode(query)

        # Compare against database, exhaustive search!
        distance_fn = distance.cosine
        similarity_scores = [(idx, distance_fn(query_embedding, row[search_column])) for idx, row in search_index.iterrows()]

        # Get Top K
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1])
        similarity_scores = similarity_scores[0:count]
        similarity_scores = filter(lambda x: x[1] <= threshold, similarity_scores)

        results = []
        for ss in similarity_scores:
            idx, score = ss
            results.append({
                'name': search_index['title'][idx],
                'url': search_index['media'][idx],
                'score': score
            })
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