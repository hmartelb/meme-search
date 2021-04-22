import os
import pandas as pd
from fastapi import FastAPI, Request

from features import TextExtractor
from scipy.spatial import distance

app = FastAPI()

INDEX_FILENAME = os.path.join('images', 'index.df')
search_index = pd.read_pickle(INDEX_FILENAME)

te = TextExtractor()

@app.get('/')
def index(query: str, count: int = 20, mode: str = 'both'):

    if mode == 'both': search_column = 'fusion_text_embedding'
    if mode == 'title': search_column = 'title_embedding'
    if mode == 'content': search_column = 'ocr_embedding'

    if query:
        # Calculate the embedding of the query
        query_embedding = te.to_vec(text=query, to_numpy=True)

        # Compare against database
        similarity_scores = [(idx, distance.cosine(query_embedding, row[search_column])) for idx, row in search_index.iterrows()]
        
        # Get Top K
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1])
        similarity_scores = similarity_scores[0:count]

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
    ap.add_argument('--gpu', default='0')
    # ap.add_argument('--debug')
    args = ap.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")#, reload=True, debug=True)
    # dev = 1
    # if dev==0:
    #     #use this one
    #     uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
    # if dev == 1:
    #     #or this one
    #     uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info", reload=True, debug=True)
    # if dev == 2:
    #     uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info", workers=2)
