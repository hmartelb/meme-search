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

@app.route('/')
def index():
    results = []
    query = request.args.get('query', None)
    count = request.args.get('count', 20)
    mode = request.args.get('mode', 'both')
    threshold = request.args.get('threshold', 0.99)

    if query is not None:
        if check_url_in_query(query):
            mode = 'url'

        if mode in ['image', 'url']:
            threshold = 100
        
        # Make API calls here
        response = requests.get(
            app.config['API_ENDPOINT'], 
            params={'query': query, 'count': count, 'mode': mode, 'threshold': threshold}
        )
        if response.status_code == 200:
            try:
                results = response.json()['results']
            except: 
                pass # Results are empty

    return render_template('pages/index.html', title='Meme search', query=query, results=results, mode=mode)

if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default=5066, type=int)
    args = ap.parse_args()

    app.config['DEBUG'] = True
    app.config['ASSETS_DEBUG'] = True
    
    port = int(os.environ.get('PORT', args.port))
    app.run(host='0.0.0.0', port=port)