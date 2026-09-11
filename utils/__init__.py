"""
Utilities Module - Shakshi Email Spam Detection Project
Created by: Shakshi
Purpose: Utility functions for model loading and prediction
"""

import os
import pickle
import logging
from config import MODEL_PATH, VECTORIZER_PATH

logger = logging.getLogger(__name__)

class ModelLoader:
    """Handle model and vectorizer loading"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
    
    def load_artifacts(self):
        """Load model and vectorizer from pickle files"""
        try:
            if not os.path.exists(MODEL_PATH):
                logger.error(f"Model file not found at {MODEL_PATH}")
                return False
            
            if not os.path.exists(VECTORIZER_PATH):
                logger.error(f"Vectorizer file not found at {VECTORIZER_PATH}")
                return False
            
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(VECTORIZER_PATH, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            logger.info("✓ Model and Vectorizer loaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"✗ Error loading artifacts: {str(e)}")
            return False
    
    def is_ready(self):
        """Check if model and vectorizer are loaded"""
        return self.model is not None and self.vectorizer is not None
    
    def get_model(self):
        """Return loaded model"""
        return self.model
    
    def get_vectorizer(self):
        """Return loaded vectorizer"""
        return self.vectorizer
