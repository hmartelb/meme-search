import os

import pandas as pd

# Configuration and constant definitions for the API

# Search
TEMPLATES_INDEX_FILENAME = 'templates.pkl'
SEARCH_INDEX_FILENAME = 'index_clean.pkl'#os.path.join('images', 'index_4.df')
SEARCH_READER_FN = pd.read_pickle
SEARCH_COLUMNS = ['fusion_text_glove', 'title_glove', 'ocr_glove', 'img_embedding']
SEARCH_MAX_DIMS = [300, 300, 300, 512]#[30,30,30,50]

# Models
PRETRAINED_MODELS_DIR = 'pretrained'
if not os.path.isdir(PRETRAINED_MODELS_DIR):
    os.makedirs(PRETRAINED_MODELS_DIR)

EMBEDDINGS_FILENAME = os.path.join(PRETRAINED_MODELS_DIR, 'glove.6B.300d_dict.pickle')
EMBEDDINGS_URL = 'https://cloud.tsinghua.edu.cn/f/0e2ab878bb5d4698b344/?dl=1'

# Temp images
ALLOWED_IMAGE_EXTENSIONS = [".jpg", ".png", ".gif"]
TEMP_IMAGES_DIR = os.path.join('images', 'external')
if not os.path.isdir(TEMP_IMAGES_DIR):
    os.makedirs(TEMP_IMAGES_DIR)