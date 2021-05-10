import os
import re

import requests
from flask import Flask, redirect, render_template, request, url_for, jsonify
from flask_assets import Bundle, Environment
from flask_caching import Cache

app = Flask(__name__)
app.config.from_object('config')
# cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})
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

@app.route('/autocomplete')
def autocomplete():
    query_text = request.args.get('query_text', None)

    if query_text is not None:
        response = requests.get(
            app.config['API_ENDPOINT'] + "/autocomplete",
            params={'query_text': query_text, 'col': 'title'}
        )
        if response.status_code == 200:
            response_json = response.json()
            return jsonify({'result': response_json })
    return jsonify({'result': []})

@app.route('/meme/<idx>')
# @cache.cached(timeout=60, query_string=True)
def meme_details(idx):
    # response = requests.get(app.config['API_ENDPOINT']+'/meme', params={'idx': idx })
    # if response.status_code == 200:
    #     meme = response.json()
    similar = []
    # Make API calls here
    response = requests.get(
        app.config['API_ENDPOINT'], 
        params={'query': idx, 'count': 5, 'mode': 'image', 'threshold': 15}
    )
    if response.status_code == 200:
        try:
            response_json = response.json()
            meme = response_json['results'][0]
            similar = response_json['results'][1:]
            valid_results = response_json['valid_results']

            return render_template(
                'pages/meme_details.html',
                title="Meme search",
                meme=meme,
                similar_memes=similar,
                valid_results=valid_results
            )
        except: 
            pass # Results are empty

    return redirect(url_for('index'))

@app.route('/templates')
# @cache.cached(timeout=60, query_string=True)
def templates():
    current_page = int(request.args.get('page', 1))
    items_per_page = int(request.args.get('count', 20))

    # Get results corresponding to the current page
    response = requests.get(
        app.config['API_ENDPOINT']+'/templates', 
        params={'page': current_page, 'items_per_page': items_per_page}
    )
    if response.status_code == 200:
        try:
            results = response.json()
            return render_template(
                'pages/templates.html', 
                title='Meme search', 
                results=results['templates'], 
                total_pages=results['total_pages'], 
                current_page=results['page'], 
                count=results['items_per_page']
            )
        except:
            pass

    return redirect(url_for('index'))

@app.route('/')
# @cache.cached(timeout=60, query_string=True)
def index():
    results = []
    query = request.args.get('query', None)
    count = request.args.get('count', 20)
    mode = request.args.get('mode', 'both')
    threshold = request.args.get('threshold', 0.99)

    valid_results = 0

    if query is not None:
        if check_url_in_query(query):
            mode = 'url'

        if mode in ['image', 'url']:
            threshold = 15
        
        if mode == 'template':
            threshold = 15

        # Make API calls here
        response = requests.get(
            app.config['API_ENDPOINT'], 
            params={'query': query, 'count': count, 'mode': mode, 'threshold': threshold}
        )
        if response.status_code == 200:
            try:
                response_json = response.json()
                results = response_json['results']
                valid_results = response_json['valid_results']
            except: 
                pass # Results are empty

    return render_template(
        'pages/index.html', 
        title='Meme search',
        query=query, 
        results=results, 
        valid_results=valid_results,
        mode=mode
    )

if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default=5006, type=int)
    ap.add_argument('--debug', default=True, type=bool)
    args = ap.parse_args()

    app.config['DEBUG'] = True
    app.config['ASSETS_DEBUG'] = True
    
    port = int(os.environ.get('PORT', args.port))
    app.run(host='0.0.0.0', port=port)