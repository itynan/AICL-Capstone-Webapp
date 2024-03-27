
# gunicorn_config.py
workers = 4
timeout = 30
bind = "0.0.0.0:7002"
errorlog = "/var/myapp/logs/"
loglevel = "info"
