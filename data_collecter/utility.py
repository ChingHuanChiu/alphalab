import logging

class Log:
    def __init__(self, log_filename, logger_name=None) -> None:
        self.log_filename = log_filename
        self.logger_name = logger_name

                
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=log_format)

        if self.logger_name is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(self.logger_name)

        handler = logging.FileHandler(self.log_filename)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)


    def write_log(self, level, msg):

        if level == "info":
            self.logger.info(msg)
        elif level == "warning":
            self.logger.warning(msg)
        elif level == "critical":
            self.logger.critical(msg)
        elif level == 'error':
            self.logger.error(msg)

