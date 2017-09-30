#!/usr/bin/env python
from scribe_classifier.flask_demo import app

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, threaded=True, use_reloader=False)
