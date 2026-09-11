"""
Logger Module - Shakshi Email Spam Detection Project
Centralized logging configuration and management
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from config import LOG_DIR, LOG_LEVEL

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

class LoggerSetup:
    """Configure and manage application logging"""
    
    def __init__(self, name='Shakshi-SpamDetection'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        
        # Log format
        log_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with rotation
        log_file = os.path.join(LOG_DIR, f'app_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(log_format)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        self.logger.addHandler(console_handler)
    
    def get_logger(self):
        """Return configured logger instance"""
        return self.logger

# Initialize logger
logger_instance = LoggerSetup()
logger = logger_instance.get_logger()
