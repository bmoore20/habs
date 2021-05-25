from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import sys
from datetime import datetime

now = datetime.now()
time_suffix = now.strftime("%Y-%m-%d_%H-%M-%S")
LOGFILE = f"/content/gdrive/MyDrive/habs_google/logs/logfile_{time_suffix}.log"
Path(LOGFILE).parent.mkdir(parents=True, exist_ok=True)

ch = logging.StreamHandler(sys.stdout)
fh = RotatingFileHandler(LOGFILE, maxBytes=2 ** 32, backupCount=1)
