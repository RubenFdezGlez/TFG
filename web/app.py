# Import necessary libraries
from flask import Flask
from flask import render_template # Import Flask library for web framework
from flask import request # Import request object for handling HTTP requests

# Comandos
# $ export FLASK_APP=app.py -- set the FLASK_APP environment variable to app.py
# $ flask run -- run the Flask application

app = Flask(__name__)

@app.route("/") # Define the root route
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True) # Run the Flask application in debug mode