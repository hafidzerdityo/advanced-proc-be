import structlog
import logging
from datetime import date, datetime
import pytz
from logging.handlers import TimedRotatingFileHandler

# configure structlog
def custom_time_stamper(_, __, event_dict):
    event_dict['timestamp'] = datetime.now(pytz.timezone('Asia/Jakarta')).isoformat()
    return event_dict

structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),        # optional, adds full stack if you want
        structlog.processors.format_exc_info,            # <-- THIS renders exception details from exc_info=True
        custom_time_stamper,
        structlog.processors.CallsiteParameterAdder(
            [structlog.processors.CallsiteParameter.FILENAME,
             structlog.processors.CallsiteParameter.FUNC_NAME]
        ),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)

# set up the log file
log_file = f"logs/app-{date.today()}.log"

# Custom formatter to adjust asctime to Indonesian time
class CustomFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, pytz.timezone('Asia/Jakarta'))
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.isoformat()
        return s

# Timed rotating file handler
file_handler = TimedRotatingFileHandler("logs/app.log", when="midnight", interval=1)
file_handler.setFormatter(CustomFormatter("%(asctime)s - %(levelname)s - %(message)s"))
file_handler.suffix = "%Y-%m-%d"  # This will append the date to the log file name

# stream handler (for terminal output)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(CustomFormatter("%(asctime)s - %(levelname)s - %(message)s"))

# configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)  # set the desired log level
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)

# get the structlog logger
logger = structlog.get_logger()
logger.info("Starting the application")

def get_logger():
    return logger