"""
Configuration module for the multi-agent customer support system.
Loads environment variables and provides centralized configuration.
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

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

# Provider & API Keys
# Supported providers: 'gemini', 'groq', 'openai' (accepts alias 'grok' -> 'groq')
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
if LLM_PROVIDER == "grok":
    LLM_PROVIDER = "groq"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate only for the selected provider (embedding usage is handled separately)
if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is required for provider 'gemini'. Set it in your .env file.")
if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is required for provider 'groq'. Set it in your .env file.")
if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required for provider 'openai'. Set it in your .env file.")

# LangSmith (optional)
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "multi-agent-customer-support")

# Model configurations
class ModelConfig:
    """Configuration for LLMs and generation params.

    Primary vs Secondary allows you to switch models per task (e.g., primary for
    Triage/Action/Escalation, secondary for Knowledge/Follow-Up) regardless of provider.
    """
    # Provider used by default text generation
    LLM_PROVIDER = LLM_PROVIDER

    # Provider-specific model names; can be overridden via env
    # Gemini defaults
    _GEMINI_PRIMARY = os.getenv("GEMINI_PRIMARY_MODEL", "gemini-2.5-pro")
    _GEMINI_SECONDARY = os.getenv("GEMINI_SECONDARY_MODEL", "gemini-2.5-flash")

    # Groq defaults: llama-3.3-70b-versatile is fast AND smart on Groq
    # For simpler tasks, use llama-3.1-8b-instant
    # Reference: https://console.groq.com/docs/models
    _GROQ_PRIMARY = os.getenv("GROQ_PRIMARY_MODEL", "llama-3.3-70b-versatile")
    _GROQ_SECONDARY = os.getenv("GROQ_SECONDARY_MODEL", "llama-3.1-8b-instant")

    # OpenAI defaults: gpt-4o for best quality, gpt-4o-mini for cost efficiency
    # Reference: https://platform.openai.com/docs/models
    _OPENAI_PRIMARY = os.getenv("OPENAI_PRIMARY_MODEL", "gpt-4o")
    _OPENAI_SECONDARY = os.getenv("OPENAI_SECONDARY_MODEL", "gpt-4o-mini")

    # Chosen models exposed generically
    if LLM_PROVIDER == "gemini":
        PRIMARY_MODEL = _GEMINI_PRIMARY
        SECONDARY_MODEL = _GEMINI_SECONDARY
    elif LLM_PROVIDER == "openai":
        PRIMARY_MODEL = _OPENAI_PRIMARY
        SECONDARY_MODEL = _OPENAI_SECONDARY
    else:  # "groq"
        PRIMARY_MODEL = _GROQ_PRIMARY
        SECONDARY_MODEL = _GROQ_SECONDARY

    # Embedding configuration (used by retrieval if available)
    # Use Gemini embeddings by default; if not available, retrieval falls back.
    EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "gemini").lower().strip()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

    # Generation parameters (lower temperature for more consistent outputs)
    TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.3"))
    MAX_OUTPUT_TOKENS = int(os.getenv("GEN_MAX_OUTPUT_TOKENS", "1024"))
    TOP_P = float(os.getenv("GEN_TOP_P", "0.9"))
    TOP_K = int(os.getenv("GEN_TOP_K", "40"))

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
    
    # Rate limit handling
    RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "2.0"))  # Delay between API calls (seconds)
    RATE_LIMIT_RETRY_DELAY = float(os.getenv("RATE_LIMIT_RETRY_DELAY", "20.0"))  # Wait time on rate limit (seconds)
    RATE_LIMIT_MAX_RETRIES = int(os.getenv("RATE_LIMIT_MAX_RETRIES", "3"))  # Max retries on rate limit

# Dataset configurations
class DatasetConfig:
    """Configuration for dataset processing"""
    BITEXT_DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    KNOWLEDGE_BASE_SIZE = 200  # Number of FAQs to include in knowledge base
    TEST_SPLIT_RATIO = 0.2  # Ratio of data for testing

print("[OK] Configuration loaded successfully")
print(f"  - Project root: {PROJECT_ROOT}")
print(f"  - Data directory: {DATA_DIR}")
print(f"  - Provider: {ModelConfig.LLM_PROVIDER}")
print(f"  - Gemini API key: {'Set' if GEMINI_API_KEY else 'Not set'}")
print(f"  - Groq API key: {'Set' if GROQ_API_KEY else 'Not set'}")
print(f"  - Primary model: {ModelConfig.PRIMARY_MODEL}")
print(f"  - Secondary model: {ModelConfig.SECONDARY_MODEL}")
print(f"  - LangSmith tracing: {'Enabled' if LANGCHAIN_TRACING_V2 else 'Disabled'}")
