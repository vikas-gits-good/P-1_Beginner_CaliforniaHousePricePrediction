import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log"
log_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)


# Custom formatter to format the line number
class CustomFormatter(logging.Formatter):
    def format(self, record):
        digits = 4
        record.lineno = f"{record.lineno:0{digits}}"  # Format line number to 4 digits
        return super().format(record)


# Set up logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)s %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Apply the custom formatter
for handler in logging.getLogger().handlers:
    handler.setFormatter(CustomFormatter(handler.formatter._fmt))
