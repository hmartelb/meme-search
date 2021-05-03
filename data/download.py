import requests


def download_image(url, filename):
    r = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(r.content)

if __name__ == '__main__':
    import argparse
    import os

    import pandas as pd
    from tqdm import tqdm
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data.tsv')
    ap.add_argument('--output', default='data_dl.tsv')
    ap.add_argument('--downloads_folder', default="images")
    ap.add_argument('--delay', default=0.5, type=float)
    args = ap.parse_args()

    if not os.path.isdir(args.downloads_folder):
        os.makedirs(args.downloads_folder)
    
    df = pd.read_csv(args.input, sep='\t')
    print(f"Total images to download {len(df)}")

    df['downloaded'] = True
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            download_image(
                url='https://'+row['url'],
                filename=os.path.join(args.downloads_folder, row['id']+row['format'])
            )
            # time.sleep(args.delay)
        except:
            df['downloaded'][i] = False
            pass

    df = df[df['downloaded'] == True]
    df.to_csv(args.output, sep='\t', index=False)