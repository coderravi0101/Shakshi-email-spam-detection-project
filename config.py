"""
Configuration Module
Centralized configuration management for the Spam Detection Application
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Environment Configuration
ENVIRONMENT = os.environ.get('FLASK_ENV', 'development')
DEBUG = ENVIRONMENT == 'development'

# Server Configuration
HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', 5000))

# Model Configuration
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')

# Logging Configuration
LOG_DIR = os.path.join(BASE_DIR, 'logs')
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# API Configuration
API_VERSION = 'v1'
API_PREFIX = f'/api/{API_VERSION}'

# Flask Configuration
class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = True

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}

def get_config():
    """Get appropriate configuration based on environment"""
    return config.get(ENVIRONMENT, DevelopmentConfig)
