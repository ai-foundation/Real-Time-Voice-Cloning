import os
from flask.logging import wsgi_errors_stream
import server

app = server.initialize()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6997)
