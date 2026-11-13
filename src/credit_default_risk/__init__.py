import logging

from .logging_conf import configure_logging


def main() -> None:
    configure_logging()
    logging.getLogger(__name__).info("Credit Default Risk command started")
