import logging

all= ['logger']

# setting the _logger format
logger = logging.getLogger('smote_variants')
logger.setLevel(logging.DEBUG)
logger_ch = logging.StreamHandler()
logger_ch.setFormatter(logging.Formatter(
    "%(asctime)s:%(levelname)s:%(message)s"))
logger.addHandler(logger_ch)
