from app import app
import waitress
# import logging

if __name__ == "__main__":
    # app.run(port=7255, debug=False)
    # logger = logging.getLogger('waitress')
    # logger.setLevel(logging.INFO)
    waitress.serve(app, listen='0.0.0.0:5006', url_scheme='https')