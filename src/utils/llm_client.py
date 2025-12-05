"""
Unified LLM client to provide a single point of access for text generation across
multiple providers. Currently supports:
- Google Gemini (google-generativeai and langchain-google-genai)
- Groq (OpenAI-compatible, e.g., Llama 3 8B)
- OpenAI (GPT-4, GPT-4o, GPT-3.5-turbo, etc.)

Usage:
    # Simple text generation:
    from utils.llm_client import get_llm_client
    client = get_llm_client()
    text = client.generate(prompt)

    # LangChain-compatible ChatModel (for agents with tool calling):
    from utils.llm_client import get_chat_model
    chat_model = get_chat_model()

Provider and model selection is controlled via .env file:
    LLM_PROVIDER=gemini|groq|openai
    GEMINI_PRIMARY_MODEL=gemini-2.0-flash
    GROQ_PRIMARY_MODEL=llama-3.1-8b-instant
    OPENAI_PRIMARY_MODEL=gpt-4o
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any

from config import ModelConfig, GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY

# Optional imports are resolved at runtime per provider
try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover - optional
    genai = None  # type: ignore

try:
    from groq import Groq  # type: ignore
except Exception:  # pragma: no cover - optional
    Groq = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional
    OpenAI = None  # type: ignore

# LangChain imports for ChatModel interface
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
except Exception:
    ChatGoogleGenerativeAI = None  # type: ignore

try:
    from langchain_groq import ChatGroq  # type: ignore
except Exception:
    ChatGroq = None  # type: ignore

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:
    ChatOpenAI = None  # type: ignore

@dataclass
class _GenParams:
    """Generation parameters for LLM calls."""
    temperature: float
    max_output_tokens: int
    top_p: float
    top_k: int


class LLMClient:
    """Provider-agnostic text generation client for simple text generation."""

    def __init__(self, provider: str, model: str, gen_params: Optional[_GenParams] = None):
        self.provider = provider.lower().strip()
        # map common typo 'grok' to 'groq'
        if self.provider == "grok":
            self.provider = "groq"
        self.model = model
        self.params = gen_params or _GenParams(
            temperature=ModelConfig.TEMPERATURE,
            max_output_tokens=ModelConfig.MAX_OUTPUT_TOKENS,
            top_p=ModelConfig.TOP_P,
            top_k=ModelConfig.TOP_K,
        )

        if self.provider == "gemini":
            if genai is None:
                raise RuntimeError("google-generativeai package not available")
            if not GEMINI_API_KEY:
                raise RuntimeError("GEMINI_API_KEY not set")
            genai.configure(api_key=GEMINI_API_KEY)
            self._client = genai.GenerativeModel(
                model_name=self.model,
                generation_config=genai.GenerationConfig(
                    temperature=self.params.temperature,
                    max_output_tokens=self.params.max_output_tokens,
                    top_p=self.params.top_p,
                    top_k=self.params.top_k,
                ),
            )
        elif self.provider == "groq":
            if Groq is None:
                raise RuntimeError("groq package not available. Add 'groq' to requirements and install.")
            if not GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY not set")
            self._client = Groq(api_key=GROQ_API_KEY)
        elif self.provider == "openai":
            if OpenAI is None:
                raise RuntimeError("openai package not available. Install with: pip install openai")
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY not set")
            self._client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text for a single-turn prompt.
        Returns str content.
        """
        if self.provider == "gemini":
            response = self._client.generate_content(prompt)
            return getattr(response, "text", "") or ""

        if self.provider == "groq":
            # OpenAI-compatible chat.completions API with retry logic and secondary model fallback
            import time
            
            # Use configured models: PRIMARY first, then SECONDARY
            models_to_try = [self.model]
            if ModelConfig.SECONDARY_MODEL and ModelConfig.SECONDARY_MODEL != self.model:
                models_to_try.append(ModelConfig.SECONDARY_MODEL)
            
            # Additional fallbacks if both configured models fail
            fallback_models = [
                "llama-3.1-70b-versatile",  # Good balance of speed and quality
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768"
            ]
            for fb in fallback_models:
                if fb not in models_to_try:
                    models_to_try.append(fb)
            
            for attempt_model in models_to_try:
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        completion = self._client.chat.completions.create(
                            model=attempt_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self.params.temperature,
                            top_p=self.params.top_p,
                            max_tokens=self.params.max_output_tokens,
                        )
                        if completion and completion.choices:
                            result = completion.choices[0].message.content or ""
                            if attempt_model != self.model:
                                print(f"[INFO] Using fallback model '{attempt_model}' (primary '{self.model}' unavailable)")
                            return result
                    except Exception as e:
                        error_msg = str(e).lower()
                        
                        # Check if it's a rate limit error
                        if "rate" in error_msg or "limit" in error_msg or "429" in error_msg:
                            if retry < max_retries - 1:
                                wait_time = 20 * (1.5 ** retry)  # 20s, 30s, 45s
                                print(f"[RATE LIMIT] Model '{attempt_model}' hit rate limit (retry {retry + 1}/{max_retries})")
                                print(f"[RATE LIMIT] Waiting {wait_time:.0f}s before retry...")
                                time.sleep(wait_time)
                                continue
                            else:
                                print(f"[RATE LIMIT] Model '{attempt_model}' rate limit exceeded after {max_retries} retries, trying next model")
                                break  # Try next model
                        
                        # Check if model is decommissioned or unavailable
                        elif "decommissioned" in error_msg or "not found" in error_msg or "does not exist" in error_msg:
                            print(f"[INFO] Model '{attempt_model}' not available, trying next model")
                            break  # Try next model immediately
                        
                        # Other errors
                        else:
                            if retry < max_retries - 1:
                                wait_time = 10 * (retry + 1)
                                print(f"[ERROR] Model '{attempt_model}' error (retry {retry + 1}/{max_retries}): {e}")
                                time.sleep(wait_time)
                                continue
                            else:
                                print(f"[ERROR] Model '{attempt_model}' failed after {max_retries} retries: {e}")
                                break  # Try next model
                
                # If we exhausted retries for this model and it's the last one, raise error
                if attempt_model == models_to_try[-1]:
                    raise RuntimeError(
                        f"All Groq models failed including fallbacks. Last model tried: {attempt_model}"
                    )
            
            return ""

        if self.provider == "openai":
            # OpenAI chat.completions API with retry logic and secondary model fallback
            import time
            
            # Use configured models: PRIMARY first, then SECONDARY
            models_to_try = [self.model]
            if ModelConfig.SECONDARY_MODEL and ModelConfig.SECONDARY_MODEL != self.model:
                models_to_try.append(ModelConfig.SECONDARY_MODEL)
            
            # Additional fallbacks if both configured models fail
            fallback_models = [
                "gpt-4o-mini",  # Fast and cost-effective
                "gpt-3.5-turbo",  # Most economical
            ]
            for fb in fallback_models:
                if fb not in models_to_try:
                    models_to_try.append(fb)
            
            for attempt_model in models_to_try:
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        completion = self._client.chat.completions.create(
                            model=attempt_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self.params.temperature,
                            top_p=self.params.top_p,
                            max_tokens=self.params.max_output_tokens,
                        )
                        if completion and completion.choices:
                            result = completion.choices[0].message.content or ""
                            if attempt_model != self.model:
                                print(f"[INFO] Using fallback model '{attempt_model}' (primary '{self.model}' unavailable)")
                            return result
                    except Exception as e:
                        error_msg = str(e).lower()
                        
                        # Check if it's a rate limit error
                        if "rate" in error_msg or "limit" in error_msg or "429" in error_msg:
                            if retry < max_retries - 1:
                                wait_time = 20 * (1.5 ** retry)  # 20s, 30s, 45s
                                print(f"[RATE LIMIT] Model '{attempt_model}' hit rate limit (retry {retry + 1}/{max_retries})")
                                print(f"[RATE LIMIT] Waiting {wait_time:.0f}s before retry...")
                                time.sleep(wait_time)
                                continue
                            else:
                                print(f"[RATE LIMIT] Model '{attempt_model}' rate limit exceeded after {max_retries} retries, trying next model")
                                break  # Try next model
                        
                        # Check if model is unavailable
                        elif "not found" in error_msg or "does not exist" in error_msg or "invalid" in error_msg:
                            print(f"[INFO] Model '{attempt_model}' not available, trying next model")
                            break  # Try next model immediately
                        
                        # Other errors
                        else:
                            if retry < max_retries - 1:
                                wait_time = 10 * (retry + 1)
                                print(f"[ERROR] Model '{attempt_model}' error (retry {retry + 1}/{max_retries}): {e}")
                                time.sleep(wait_time)
                                continue
                            else:
                                print(f"[ERROR] Model '{attempt_model}' failed after {max_retries} retries: {e}")
                                break  # Try next model
                
                # If we exhausted retries for this model and it's the last one, raise error
                if attempt_model == models_to_try[-1]:
                    raise RuntimeError(
                        f"All OpenAI models failed including fallbacks. Last model tried: {attempt_model}"
                    )
            
            return ""

        raise ValueError(f"Unsupported provider: {self.provider}")


def get_chat_model(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    provider: Optional[str] = None
) -> Any:
    """
    Get a LangChain-compatible ChatModel for use with agents and tool calling.
    
    Args:
        model_name: Model name to use. If None, uses PRIMARY_MODEL from config.
        temperature: Temperature for generation. If None, uses config default.
        provider: Provider to use ('gemini', 'groq', or 'openai'). If None, uses config default.
    
    Returns:
        LangChain ChatModel (ChatGoogleGenerativeAI, ChatGroq, or ChatOpenAI)
    
    Example:
        chat_model = get_chat_model()
        chat_model = get_chat_model(model_name="gpt-4o", temperature=0.3, provider="openai")
    """
    _provider = (provider or ModelConfig.LLM_PROVIDER).lower().strip()
    if _provider == "grok":
        _provider = "groq"
    
    _model = model_name or ModelConfig.PRIMARY_MODEL
    _temperature = temperature if temperature is not None else ModelConfig.TEMPERATURE
    
    if _provider == "gemini":
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError(
                "langchain-google-genai package not available. "
                "Install with: pip install langchain-google-genai"
            )
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set in .env file")
        
        return ChatGoogleGenerativeAI(
            model=_model,
            google_api_key=GEMINI_API_KEY,
            temperature=_temperature,
            max_output_tokens=ModelConfig.MAX_OUTPUT_TOKENS,
            convert_system_message_to_human=True  # Required for Gemini
        )
    
    elif _provider == "groq":
        if ChatGroq is None:
            raise RuntimeError(
                "langchain-groq package not available. "
                "Install with: pip install langchain-groq"
            )
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set in .env file")
        
        return ChatGroq(
            model=_model,
            api_key=GROQ_API_KEY,
            temperature=_temperature,
            max_tokens=ModelConfig.MAX_OUTPUT_TOKENS,
        )
    
    elif _provider == "openai":
        if ChatOpenAI is None:
            raise RuntimeError(
                "langchain-openai package not available. "
                "Install with: pip install langchain-openai"
            )
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set in .env file")
        
        return ChatOpenAI(
            model=_model,
            api_key=OPENAI_API_KEY,
            temperature=_temperature,
            max_tokens=ModelConfig.MAX_OUTPUT_TOKENS,
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {_provider}. Use 'gemini', 'groq', or 'openai'.")


def get_llm_client(model_name: Optional[str] = None) -> LLMClient:
    """
    Get an LLMClient instance for simple text generation.
    
    Args:
        model_name: Model name to use. If None, uses PRIMARY_MODEL from config.
    
    Returns:
        LLMClient instance
    
    Example:
        client = get_llm_client()
        response = client.generate("Hello, world!")
    """
    provider = ModelConfig.LLM_PROVIDER
    model = model_name or ModelConfig.PRIMARY_MODEL
    return LLMClient(provider=provider, model=model)


# Aliases for backward compatibility
build_llm_client = get_llm_client


def get_embedding_function():
    """
    Get embedding function based on config.
    Currently only supports Gemini embeddings.
    
    Returns:
        Callable that generates embeddings
    """
    if ModelConfig.EMBEDDINGS_PROVIDER == "gemini":
        if genai is None:
            raise RuntimeError("google-generativeai package not available")
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=GEMINI_API_KEY)
        
        def embed_content(text: str, task_type: str = "retrieval_query") -> list:
            """Generate embedding for text using Gemini."""
            result = genai.embed_content(
                model=ModelConfig.EMBEDDING_MODEL,
                content=text,
                task_type=task_type
            )
            return result['embedding']
        
        return embed_content
    else:
        raise ValueError(f"Unsupported embedding provider: {ModelConfig.EMBEDDINGS_PROVIDER}")
