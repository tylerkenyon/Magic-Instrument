from __future__ import annotations

import logging
import sys

from colorama import Fore, Style, init


init(autoreset=True)


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        timestamp = self.formatTime(record, self.datefmt)
        prefix = f"{color}[{record.levelname:<8}]{Style.RESET_ALL}"
        return f"{prefix} {timestamp} | {record.getMessage()}"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("invisible_instrument")
    if logger.handlers:
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter(datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
