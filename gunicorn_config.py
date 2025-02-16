bind = "0.0.0.0:10000"
workers = 1
threads = 1
timeout = 300
worker_class = "gthread"
max_requests = 5
max_requests_jitter = 2
worker_tmp_dir = "/dev/shm"
preload_app = True
# Reduce memory usage further
worker_connections = 50
keepalive = 2 