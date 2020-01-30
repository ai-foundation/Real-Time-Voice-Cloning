import multiprocessing
import os

# def _get_thread_count():


# Gunicorn settings
host = os.getenv("HOST", "0.0.0.0")
port = os.getenv("PORT", "6009")
bind = f"{host}:{port}"
# suggested number of workers + threads: (2*CPU)+1
workers = 1
worker_class = "gthread"
threads = 1


