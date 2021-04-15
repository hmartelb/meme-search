import os

import requests
from flask import Flask, redirect, render_template, request, url_for
from flask_assets import Bundle, Environment

app = Flask(__name__)
app.config.from_object('config')

# assets = Environment(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print('query', request.args.get('query',' '))
        return render_template('pages/index.html', title='Meme src')
    else:
        return render_template('pages/index.html', title='Meme search')


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default=5066, type=int)
    args = ap.parse_args()

    app.config['DEBUG'] = True
    app.config['ASSETS_DEBUG'] = True
    
    port = int(os.environ.get('PORT', args.port))
    app.run(host='0.0.0.0', port=port)