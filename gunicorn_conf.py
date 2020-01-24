import multiprocessing
import os

# def _get_thread_count():


# Gunicorn settings
host = os.getenv("HOST", "0.0.0.0")
port = os.getenv("PORT", "5009")
bind = f"{host}:{port}"
workers = 1
worker_class = "gthread"
threads = 2


