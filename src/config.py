"""
Configuration module for the multi-agent customer support system.
Loads environment variables and provides centralized configuration.
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(override=True)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
KNOWLEDGE_BASE_FILE = DATA_DIR / "knowledge_base.json"
DATASET_FILE = DATA_DIR / "bitext_dataset.json"
TEST_DATASET_FILE = DATA_DIR / "test_conversations.json"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required. Please set it in .env file")

# LangSmith (optional)
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "multi-agent-customer-support")

# Model configurations
class ModelConfig:
    """Configuration for Gemini models"""
    # Primary model for critical tasks (Triage, Action, Escalation)
    GEMINI_PRO = "gemini-2.5-pro"

    # Fast model for high-volume tasks (Knowledge, Follow-Up)
    GEMINI_FLASH = "gemini-2.5-flash"

    # Embedding model for vector store
    EMBEDDING_MODEL = "models/text-embedding-004"

    # Generation parameters
    TEMPERATURE = 0.7
    MAX_OUTPUT_TOKENS = 2048
    TOP_P = 0.95
    TOP_K = 40

# Agent configurations
class AgentConfig:
    """Configuration for agent behavior"""
    MAX_MEMORY_MESSAGES = 10  # Number of messages to keep in session memory
    CHROMA_COLLECTION_NAME = "customer_support_kb"
    RETRIEVAL_TOP_K = 5  # Number of documents to retrieve from vector store

# Evaluation configurations
class EvalConfig:
    """Configuration for evaluation metrics"""
    TEST_SAMPLE_SIZE = 100  # Number of conversations for evaluation
    CSAT_SCALE = (1, 5)  # CSAT score range
    CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for automated resolution
    RATE_LIMIT_DELAY = 20  # Seconds to wait between requests to avoid rate limits

# Dataset configurations
class DatasetConfig:
    """Configuration for dataset processing"""
    BITEXT_DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    KNOWLEDGE_BASE_SIZE = 200  # Number of FAQs to include in knowledge base
    TEST_SPLIT_RATIO = 0.2  # Ratio of data for testing

print("[OK] Configuration loaded successfully")
print(f"  - Project root: {PROJECT_ROOT}")
print(f"  - Data directory: {DATA_DIR}")
print(f"  - Gemini API key: {'Set' if GEMINI_API_KEY else 'Not set'}")
print(f"  - LangSmith tracing: {'Enabled' if LANGCHAIN_TRACING_V2 else 'Disabled'}")
