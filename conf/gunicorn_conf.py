import gevent.monkey
gevent.monkey.patch_all()
import multiprocessing

bind = '0.0.0.0:80'

# reload = True
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'gevent'

x_forwarded_for_header = 'X-FORWARDED-FOR'