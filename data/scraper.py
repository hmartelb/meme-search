'''
https://gist.github.com/WalterSimoncini/defca6de456bb168ada303085358bf0a
'''

import json
import time
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

def process_category(args):
    fetched_memes = []
    errors = 0
    for i in tqdm(range(args.from_page, args.pages + 1)):
        # print(f"Processing page {i}")
        response = requests.get(f"{args.source_url}?page={i}")
        body = BeautifulSoup(response.text, 'html.parser')

        if response.status_code != 200:
            # print("Something went wrong!")
            break # Something went wrong (e.g. page limit)

        memes = body.findAll("div", {"class": "base-unit clearfix"})
        for meme in memes:
            if "not-safe-for-work images" in str(meme):
                continue  # NSFW memes are available only to logged in users
            
            try:
                meme_url = meme.find("img", {"class": "base-img"})["src"][2:]
                meme_id, meme_format = os.path.splitext(meme_url.split("/")[-1])

                # Handle anonymous authors
                meme_author = meme.find("a", {"class": "u-username"})
                meme_author = "anonymous" if meme_author is None else meme_author.text
                
                # Handle empty titles
                meme_title = meme.find("h2", {"class": "base-unit-title"}).find("a")
                meme_title = "" if meme_title is None else meme_title.text
                    
                meme_text = meme.find("img", {"class": "base-img"})["alt"]
                meme_text = meme_text.split("|")[1].strip()

                meme_data = {
                    "id": meme_id,
                    "format": meme_format,
                    "website": "imgflip",
                    "url": 'https://'+meme_url,
                    "author": meme_author,
                    "title": meme_title,
                    "text": meme_text.lower()
                }
                fetched_memes.append(meme_data)
            except:
                errors += 1

        time.sleep(args.delay)

    print(f"Fetched: {len(fetched_memes)} memes. Found {errors} error(s).")
    return fetched_memes

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    # ap.add_argument("--source_url", default="https://imgflip.com/tag/programming", help="Memes list url (e.g. https://imgflip.com/meme/Bird-Box)", type=str)
    ap.add_argument("--tag", required=True, type=str)#default=['programming', 'artificial intelligence', 'computer'], type=list)
    ap.add_argument("--from_page", default=1, help="Initial page", type=int)
    ap.add_argument("--pages", default=10, help="Maximum page number to be scraped", type=int)
    ap.add_argument("--delay", default=2, help="Delay between page loads (seconds)", type=int)
    ap.add_argument("-o", "--output", default="data.tsv")
    args = ap.parse_args()

    # category = args.source_url.split("/")[-1].replace("-", " ")

    # Get the data
    data = {}
    # for tag in args.tags:
    print(f"Processing tag: {args.tag}")
    
    # Get the data
    args.source_url = f"https://imgflip.com/tag/{args.tag.replace(' ', '+')}"
    data = process_category(args)
    
    # Create a pd.DataFrame and save (append to existing .tsv)
    df = pd.DataFrame(data)
    print(df.head(20))
    df.to_csv(args.output, sep='\t', index=False, mode='a')