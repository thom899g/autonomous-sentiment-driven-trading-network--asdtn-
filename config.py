"""
ASDTN Configuration Manager
Handles environment variables, logging setup, and Firebase configuration
"""
import os
import logging
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, firestore

@dataclass
class TradingConfig:
    """Centralized configuration for trading parameters"""
    # Risk management
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio
    STOP_LOSS_PERCENT: float = 0.02  # 2% stop loss
    TAKE_PROFIT_PERCENT: float = 0.05  # 5% take profit
    
    # Trading hours (UTC)
    TRADING_HOURS_START: int = 13  # 9 AM EST
    TRADING_HOURS_END: int = 20    # 4 PM EST
    
    # Model parameters
    SENTIMENT_WEIGHT: float = 0.3
    TECHNICAL_WEIGHT: float = 0.7
    MIN_CONFIDENCE_THRESHOLD: float = 0.65

class ConfigManager:
    """Manages configuration and Firebase initialization"""
    
    def __init__(self):
        self.config = TradingConfig()
        self.firebase_app = None
        self.firestore_client = None
        self._setup_logging()
        self._init_firebase()
        
    def _setup_logging(self) -> None:
        """Configure structured logging for all modules"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('asdtn.log', mode='a')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _init_firebase(self) -> None:
        """Initialize Firebase with error handling"""
        try:
            # Check for Firebase credentials
            cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'firebase_credentials.json')
            
            if not Path(cred_path).exists():
                self.logger.warning(f"Firebase credentials not found at {cred_path}")
                # Create placeholder for development (will fail on actual Firebase calls)
                return
                
            # Initialize Firebase
            cred = credentials.Certificate(cred_path)
            self.firebase_app = firebase_admin.initialize_app(cred)
            self.firestore_client = firestore.client()
            self.logger.info("Firebase initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Firebase initialization failed: {str(e)}")
            # System can run in limited mode without Firebase
            self.logger.warning("Running in local-only mode without Firebase")
            
    def get_market_config(self, symbol: str) -> Dict[str, Any]:
        """Retrieve market-specific configuration from Firebase or defaults"""
        default_config = {
            'tick_size': 0.01,
            'min_order_size': 1.0,
            'max_leverage': 3.0,
            'supported_exchanges': ['binance', 'coinbase']
        }
        
        if self.firestore_client:
            try:
                doc_ref = self.firestore_client.collection('market_config').document(symbol)
                doc = doc_ref.get()
                if doc.exists:
                    return {**default_config, **doc.to_dict()}
            except Exception as e:
                self.logger.error(f"Failed to fetch market config for {symbol}: {str(e)}")
                
        return default_config

# Global configuration instance
config_manager = ConfigManager()