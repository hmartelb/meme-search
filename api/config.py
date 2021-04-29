import os

import pandas as pd

# Configuration and constant definitions for the API

# Search
SEARCH_INDEX_FILENAME = os.path.join('images', 'index_4.df')
SEARCH_READER_FN = pd.read_pickle
SEARCH_COLUMNS = ['fusion_text_glove', 'title_glove', 'ocr_glove', 'img_embedding']
SEARCH_MAX_DIM = 512

# Models
EMBEDDINGS_FILENAME = os.path.join('pretrained', 'glove.6B.300d_dict.pickle')
