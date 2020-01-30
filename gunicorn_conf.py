import multiprocessing
import os

# def _get_thread_count():


# Gunicorn settings
host = os.getenv("HOST", "0.0.0.0")
port = os.getenv("PORT", "5009")
bind = f"{host}:{port}"
# number of workers and threads are bound by number of gpu and gpu size
workers = 1
worker_class = "gthread"
threads = 1


