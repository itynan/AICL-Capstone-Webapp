
# gunicorn_config.py
workers = 4
timeout = 30
bind = "0.0.0.0:7003"
errorlog = "/var/aicyberlabs/logs/error.log"
loglevel = "info"
