# Import necessary libraries
from flask import Flask
from flask import render_template # Import Flask library for web framework
from flask import request # Import request object for handling HTTP requests

# Comandos
# $ export FLASK_APP=app.py -- set the FLASK_APP environment variable to app.py
# $ flask run -- run the Flask application

app = Flask(__name__)

