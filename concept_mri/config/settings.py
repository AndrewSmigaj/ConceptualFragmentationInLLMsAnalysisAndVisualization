"""
UI-specific settings for Concept MRI application.
Core analysis settings are imported from parent project.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
UPLOAD_DIR = DATA_DIR / "uploads"
EXPORT_DIR = DATA_DIR / "exports"

# Ensure directories exist
for dir_path in [DATA_DIR, CACHE_DIR, UPLOAD_DIR, EXPORT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# App metadata
APP_NAME = "Concept MRI"
APP_VERSION = "1.0.0"
APP_SUBTITLE = "Neural Network Analysis Tool"

# Server settings
DEBUG = os.getenv('DASH_DEBUG', 'True').lower() == 'true'
PORT = int(os.getenv('PORT', 8050))
HOST = os.getenv('HOST', '127.0.0.1')

# Upload settings
MAX_UPLOAD_SIZE_MB = 500
UPLOAD_FOLDER_ROOT = str(UPLOAD_DIR)
ALLOWED_MODEL_EXTENSIONS = ['.pt', '.pth', '.onnx', '.h5', '.pb', '.pkl']
ALLOWED_DATA_EXTENSIONS = ['.csv', '.npz', '.pkl', '.json', '.parquet']

# UI settings
THEME_COLOR = "#00A6A6"  # Medical teal
SECONDARY_COLOR = "#0066CC"  # Clinical blue
BACKGROUND_COLOR = "#FFFFFF"
ACCENT_COLOR = "#FF6B35"
TEXT_COLOR = "#1A1A1A"

# Visualization defaults
DEFAULT_TOP_N_PATHS = 25
DEFAULT_SANKEY_HEIGHT = 700
DEFAULT_SANKEY_WIDTH = "100%"
DEFAULT_CARD_HEIGHT = 200
MAX_CLUSTERS_DISPLAY = 20

# Cache settings
CACHE_TYPE = 'filesystem'
CACHE_DEFAULT_TIMEOUT = 3600  # 1 hour
CACHE_THRESHOLD = 100  # Max number of items

# Session settings
SESSION_TYPE = 'filesystem'
SESSION_FILE_DIR = str(CACHE_DIR / "sessions")
SESSION_PERMANENT = False
SESSION_USE_SIGNER = True
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# LLM settings (UI defaults)
DEFAULT_LLM_PROVIDER = 'openai'
LLM_PROVIDERS = {
    'openai': {'name': 'OpenAI', 'models': ['gpt-4', 'gpt-3.5-turbo']},
    'anthropic': {'name': 'Anthropic', 'models': ['claude-3-opus', 'claude-3-sonnet']},
    'google': {'name': 'Google', 'models': ['gemini-pro']},
    'local': {'name': 'Local Model', 'models': ['llama2', 'mistral']}
}

# Clustering UI defaults
DEFAULT_CLUSTERING_ALGORITHM = 'kmeans'
DEFAULT_METRIC = 'gap'
CLUSTERING_ALGORITHMS = ['kmeans', 'dbscan']
CLUSTERING_METRICS = ['gap', 'silhouette', 'elbow']
MIN_CLUSTERS = 2
MAX_CLUSTERS = 20

# Progress messages
MESSAGES = {
    'upload_start': 'Uploading file...',
    'upload_complete': 'Upload complete!',
    'model_loading': 'Loading model architecture...',
    'activation_extraction': 'Extracting activations...',
    'clustering_start': 'Performing clustering analysis...',
    'clustering_complete': 'Clustering complete!',
    'llm_analysis': 'Generating insights with LLM...',
    'visualization_ready': 'Visualization ready!'
}

# Import paths to parent project
import sys
PARENT_DIR = Path(__file__).parent.parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))