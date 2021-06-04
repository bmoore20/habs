import sys
import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from torch.utils.tensorboard import SummaryWriter

now = datetime.now()
time_suffix = now.strftime("%Y-%m-%d_%H-%M-%S")

# logging
LOGFILE = f"/content/gdrive/MyDrive/habs_google/logs/logfile_{time_suffix}.log"
Path(LOGFILE).parent.mkdir(parents=True, exist_ok=True)

ch = logging.StreamHandler(sys.stdout)
fh = RotatingFileHandler(LOGFILE, maxBytes=2 ** 32, backupCount=1)

# tensorboard
TENSORBOARD_DIR = "/content/gdrive/MyDrive/habs_google/runs"
RUN_DIR = str(Path(TENSORBOARD_DIR) / time_suffix)

sw = SummaryWriter(log_dir=RUN_DIR)
