import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from features import ImageExtractor, similarity_matrix


def remove_duplicate_images(df, threshold=0.5):
    ie = ImageExtractor()

    # Extract the features of each row
    df['img_embedding'] = None
    df['img_embedding_norm'] = None
    for i, row in tqdm(df.iterrows(), total=len(df)):
        df['img_embedding'].iloc[i] = ie.to_vec(
            filename=os.path.join(args.downloads_folder, row['id']+row['format']), 
            to_numpy=True
        )
        df['img_embedding_norm'].iloc[i] = df['img_embedding'].iloc[i] / np.linalg.norm(df['img_embedding'].iloc[i])

    vectors = df['img_embedding_norm'].to_numpy()
    M = similarity_matrix(vectors)
    # import matplotlib.pyplot as plt
    # plt.matshow(M)
    # plt.show()
    duplicate_pairs = []
    for i in range(len(M)):
        for j in range(i+1, len(M)):
            if M[i,j] < threshold:
                duplicate_pairs.append((i,j))
    # print(f"Found {len(duplicate_pairs)} duplicate pairs")
    df = df.drop(index=set([j for i,j in duplicate_pairs]))
    df = df.drop(columns=['img_embedding_norm'])
    return df

def remove_duplicate_text(df):
    df = df.drop_duplicates(subset=['title', 'text'], keep='first')
    return df

if __name__ == '__main__':
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--input', default='templates.tsv')
    ap.add_argument('-o','--output', default='templates.pkl')
    ap.add_argument('--mode', default='image', choices=['text', 'image'])
    ap.add_argument('--downloads_folder', default='templates')
    ap.add_argument('--threshold', default=0.5, type=float)
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep='\t')
    # df = df.iloc[0:100]

    print(f"Original length: {len(df)}")

    if args.mode == 'text':
        df = remove_duplicate_text(df)
    else:
        df = remove_duplicate_images(df, threshold=args.threshold)
    
    print(df.head(10))

    df.to_pickle(args.output)
    print(f"Length without duplicates: {len(df)}")
    print(f"Saved to {args.output}")