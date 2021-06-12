  
import os

# Grabs the folder where the script runs.
basedir = os.path.abspath(os.path.dirname(__file__))

# Enable debug mode ?
DEBUG = False
ASSETS_DEBUG = False

# Secret key for session management
SECRET_KEY = os.urandom(24)

# Connect to the database
# SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'database.db')

API_ENDPOINT = 'http://localhost:10004'