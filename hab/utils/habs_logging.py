from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import sys

LOGFILE = "./log.log"
Path(LOGFILE).parent.mkdir(parents=True, exist_ok=True)

ch = logging.StreamHandler(sys.stdout)
fh = RotatingFileHandler(LOGFILE, maxBytes=2**32, backupCount=1)
