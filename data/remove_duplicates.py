import os

import numpy as np
import pandas as pd

from features import ImageExtractor, similarity_matrix

if __name__ == '__main__':
    import argparse

    from tqdm import tqdm
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--input', default='templates.tsv')
    ap.add_argument('-o','--output', default='templates.pkl')
    ap.add_argument('--downloads_folder', default='templates')
    ap.add_argument('--threshold', default=0.5, type=float)
    args = ap.parse_args()

    ie = ImageExtractor()
    df = pd.read_csv(args.input, sep='\t')

    # df = df.iloc[0:100]

    df['img_embedding'] = None
    df['img_embedding_norm'] = None
    for i, row in tqdm(df.iterrows(), total=len(df)):
        df['img_embedding'].iloc[i] = ie.to_vec(
            filename=os.path.join(args.downloads_folder, row['id']+row['format']), 
            to_numpy=True
        )
        df['img_embedding_norm'].iloc[i] = df['img_embedding'].iloc[i] / np.linalg.norm(df['img_embedding'].iloc[i])
    print(df.head(10))

    vectors = df['img_embedding_norm'].to_numpy()
    print(vectors.shape)

    import matplotlib.pyplot as plt

    M = similarity_matrix(vectors)
    # plt.matshow(M)
    # plt.show()

    threshold = args.threshold
    duplicate_pairs = []
    for i in range(len(M)):
        for j in range(i+1, len(M)):
            if M[i,j] < threshold:
                duplicate_pairs.append((i,j))

    print(f"Found {len(duplicate_pairs)} duplicate pairs")
    # for pair in duplicate_pairs:
    #     i,j = pair
    #     img_i = os.path.join(args.downloads_folder, df['id'].iloc[i]+df['format'].iloc[i])
    #     img_j = os.path.join(args.downloads_folder, df['id'].iloc[j]+df['format'].iloc[j])
    #     print(img_i, img_j)
        
        # Propagate the title information in j to i before removing in case they are not the same ??
        # if df['title'].iloc[i] != df['title'].iloc[j]:
        #     df['title'].iloc[i] += ' ' + df['title'].iloc[j] 

    df = df.drop(index=set([j for i,j in duplicate_pairs]))
    df = df.drop(columns=['img_embedding_norm'])
    
    df.to_pickle(args.output)
    print(f"Templates without duplicates: {len(df)}. Saved to {args.output}")