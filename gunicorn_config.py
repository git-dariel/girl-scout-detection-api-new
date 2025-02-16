bind = "0.0.0.0:10000"
workers = 1
threads = 1
timeout = 300
worker_class = "gthread"
max_requests = 10
max_requests_jitter = 3
worker_tmp_dir = "/dev/shm"
preload_app = True
# Reduce memory usage
worker_connections = 100
keepalive = 5 