import logging

__author__ = 'pliskowski'

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
# ch.setLevel(logging.ERROR)

# create formatter and add it to the handlers
formatter = logging.Formatter(
    '%(asctime)s.%(msecs)03d [%(levelname)s] [%(threadName)s] (%(filename)s:%(lineno)d) -- %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(ch)