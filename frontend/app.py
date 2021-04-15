import os
import re

import requests
from flask import Flask, redirect, render_template, request, url_for
from flask_assets import Bundle, Environment

app = Flask(__name__)
app.config.from_object('config')

# assets = Environment(app)

def check_url_in_query(text):
    '''
    Modified the method to just return whether or not there is an url
    https://www.geeksforgeeks.org/python-check-url-string/
    '''
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    urls = re.findall(regex, text)      
    urls = [x[0] for x in urls]
    return len(urls) > 0

def get_default_results():
    r = {
        'name': "Meme", 
        'url': "https://pbs.twimg.com/media/EWmtqxnXgAAZUFa.jpg"
    }
    return [r for _ in range(100)]

@app.route('/')
def index():
    results = []
    query = request.args.get('query', None)
    if query is not None:
        # Make API calls here
        if check_url_in_query(query):
            # Retrieve image from url, extract features and search 
            print("Is URL: ", query)
        else:
            # Search using text input
            print("Normal text:", query)
        results = get_default_results()

    return render_template('pages/index.html', title='Meme search', query=query, results=results)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default=5066, type=int)
    args = ap.parse_args()

    app.config['DEBUG'] = True
    app.config['ASSETS_DEBUG'] = True
    
    port = int(os.environ.get('PORT', args.port))
    app.run(host='0.0.0.0', port=port)