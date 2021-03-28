from pathlib import Path
import logging
import sys

LOGFILE = "./log.log"
Path(LOGFILE).parent.mkdir(parents=True, exist_ok=True)

ch = logging.StreamHandler(sys.stdout)
fh = logging.handlers.RotatingFileHandler(LOGFILE, maxBytes=10000, backupCount=1)

# QUESTION: Does RFH delete oldest entry or create new file? Reference below says new file.
# REFERENCE: https://docs.python.org/3/library/logging.handlers.html#logging.handlers.RotatingFileHandler
# QUESTION: What is a good size for maxBytes?
