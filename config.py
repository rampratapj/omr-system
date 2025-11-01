"""
Configuration settings for OMR Evaluation System
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
DB_FILE = os.path.join(BASE_DIR, "omr_database.db")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# File settings
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'pdf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Image processing settings
THRESHOLD_VALUE = 127
Y_THRESHOLD = 30  # Distance between rows
DARKNESS_THRESHOLD = 50    # Minimum darkness to mark as filled

# Bubble detection
BUBBLE_AREA_MIN = 40
BUBBLE_AREA_MAX = 10000
ASPECT_RATIO_MIN = 0.5
ASPECT_RATIO_MAX = 1.6
MIN_BUBBLES = 20
CLARITY_THRESHOLD = 50

# Flask settings
DEBUG = True
SECRET_KEY = 'your-secret-key-change-in-production'
SESSION_TIMEOUT = 1800  # 30 minutes

# Logging
LOG_FILE = os.path.join(BASE_DIR, 'omr_system.log')
LOG_LEVEL = 'INFO'

SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_SAMESITE = 'Lax'
