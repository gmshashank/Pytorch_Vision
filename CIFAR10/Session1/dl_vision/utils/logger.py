import logging
import sys

log_level = logging.INFO


def setup_logger(name):

    # logging.basicConfig(filename="logs/dl_vision.log", filemode="a")
    logging.info("Running DL Vision")

    logger = logging.getLogger(f"dl_vision - {name}")

    # set the Log level
    logger.setLevel(log_level)

    logger_format = logging.Formatter(
        "[ %(asctime)s -%(name)s ] %(levelname)s: %(message)s"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logger_format)
    logger.addHandler(stream_handler)
    logger.propagate = False

    return logger
