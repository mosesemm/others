import logging

logging.basicConfig(level = logging.DEBUG,
                    filemode="w",
                    filename="./app.log",
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

variable = "some error"

logger.error("test one two: %s", variable)


a = 5
b = 0

try:
    c = a/b
except Exception as e:
    #logger.error("Exception occured", exc_info=True)
    logger.exception("Exception occured")

logger.debug("This is a debug message")
logger.info("this is an info message")
logger.warning("this is a warning message")
logger.error("this is an error message")
logger.critical("this is a critical message")