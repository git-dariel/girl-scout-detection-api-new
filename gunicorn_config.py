# Gunicorn configuration file
bind = "0.0.0.0:10000"
workers = 1  # Reduce number of workers for free tier
worker_class = "sync"
threads = 2
timeout = 120  # Increased timeout for processing
max_requests = 10
max_requests_jitter = 5
preload_app = True
worker_tmp_dir = "/dev/shm"  # Use RAM-based temporary directory
# Reduce memory usage further
worker_connections = 50
keepalive = 2 