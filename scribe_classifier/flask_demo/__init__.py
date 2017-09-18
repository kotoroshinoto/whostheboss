from flask import Flask
app = Flask(__name__)
from scribe_classifier.flask_demo import views
