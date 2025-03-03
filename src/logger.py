#### This is the logger file
import os
import sys
import logging
from datetime import datetime

logging_str="[%(asctime)s : %(levelname)s: %(module)s: %(message)s]"
log_dir="logs"

log_filepath=os.path.join(log_dir,f"log_{datetime.now().strftime('%Y-%m-%d_%H')}.log")

os.makedirs(log_dir,exist_ok=True)
# Suppress Great Expectations INFO logs
logging.getLogger("great_expectations").setLevel(logging.WARNING)
# Suppress feast INFO logs
logging.getLogger("feast").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath), #this will write the log into the log file
        logging.StreamHandler(sys.stdout) #this will print the message on console
    ]
)

logger=logging.getLogger("CustomLogger")