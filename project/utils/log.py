import logging


def setup_logger(name):
    """
    Configures and returns a logger with the given name.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Set the logging level

    # Create handlers (e.g., console, file handler)
    c_handler = logging.StreamHandler()  # Console handler
    f_handler = logging.FileHandler("project.log")  # File handler
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add them to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    if not logger.handlers:  # Avoid adding handlers multiple times
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger
