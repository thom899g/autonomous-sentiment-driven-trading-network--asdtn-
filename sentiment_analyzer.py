"""
Real-time Sentiment Analysis Module
Processes news and social media data using VADER and BERT models
"""
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    """Advanced sentiment analysis with multiple models and real-time processing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vader = SentimentIntensityAnalyzer()
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Cache for sentiment scores to avoid redundant processing
        self.sentiment_cache = {}
        self.cache_expiry = timedelta(minutes=5)
        
        # News sources configuration
        self.news_sources = {
            'reuters': 'https://newsapi.org/v2/everything?domains=reuters.com&apiKey={}',
            'bloomberg': 'https://newsapi.org/v2/everything?domains=bloomberg.com&apiKey={}'
        }
        
        self.logger.info("SentimentAnalyzer initialized with VADER and TextBlob")
        
    def analyze_text(self, text: str, source: str = 'unknown') -> Dict[str, float]:
        """
        Analyze sentiment of a single text with multiple models
        
        Args:
            text: Text to analyze
            source: Source of the text for logging
            
        Returns:
            Dictionary with sentiment scores and confidence
        """
        try:
            if not text or len(text.strip()) < 10:
                self.logger.warning(f"Invalid text from {source}: too short")
                return self._get_neutral_sentiment()
            
            # Check cache
            cache_key = hash(text[:100])  # First 100 chars for cache key
            if cache_key in self.sentiment_cache:
                cached_time, cached_result = self.sentiment_cache[cache_key]
                if datetime.now() - cached_time < self.cache_expiry:
                    return cached_result
            
            # VADER sentiment (optimized for social media)
            vader_scores = self.vader.polarity_scores(text)
            
            # TextBlob sentiment
            blob = TextBlob(text)