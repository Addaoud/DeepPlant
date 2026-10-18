import logging
import os


def configure_logging_format(file_path):
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    if logging.getLogger(__name__).hasHandlers():
        return logging.getLogger(__name__)

    console_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    console_handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    fileHandler = logging.FileHandler(os.path.join(file_path, "info.log"))
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger
