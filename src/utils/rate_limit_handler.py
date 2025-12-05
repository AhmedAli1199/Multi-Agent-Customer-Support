"""
Rate limit handler for LLM API calls.
Provides retry logic with exponential backoff for rate-limited requests.
"""
import time
import functools
from typing import Callable, Any, Optional
from config import EvalConfig, ModelConfig


class RateLimitError(Exception):
    """Exception raised when rate limit is hit after all retries."""
    pass


def is_rate_limit_error(error: Exception) -> bool:
    """Check if an error is a rate limit error."""
    error_str = str(error).lower()
    rate_limit_indicators = [
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        "429",
        "quota exceeded",
        "resource exhausted",
        "requests per minute",
        "rpm",
        "tokens per minute",
        "tpm"
    ]
    return any(indicator in error_str for indicator in rate_limit_indicators)


def with_rate_limit_retry(
    max_retries: Optional[int] = None,
    retry_delay: Optional[float] = None,
    delay_between_calls: Optional[float] = None
):
    """
    Decorator to add rate limit retry logic to a function.
    
    Args:
        max_retries: Maximum number of retries (default: EvalConfig.RATE_LIMIT_MAX_RETRIES)
        retry_delay: Delay in seconds between retries (default: EvalConfig.RATE_LIMIT_RETRY_DELAY)
        delay_between_calls: Delay after successful call (default: EvalConfig.RATE_LIMIT_DELAY)
    
    Usage:
        @with_rate_limit_retry()
        def my_llm_call():
            return llm.generate("prompt")
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            _max_retries = max_retries if max_retries is not None else EvalConfig.RATE_LIMIT_MAX_RETRIES
            _retry_delay = retry_delay if retry_delay is not None else EvalConfig.RATE_LIMIT_RETRY_DELAY
            _delay_between = delay_between_calls if delay_between_calls is not None else EvalConfig.RATE_LIMIT_DELAY
            
            last_error = None
            
            for attempt in range(_max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # Add delay after successful call to prevent hitting rate limits
                    if _delay_between > 0:
                        time.sleep(_delay_between)
                    
                    return result
                    
                except Exception as e:
                    last_error = e
                    
                    if is_rate_limit_error(e):
                        if attempt < _max_retries:
                            # Exponential backoff: increase delay with each retry
                            wait_time = _retry_delay * (1.5 ** attempt)
                            print(f"\n[RATE LIMIT] Hit rate limit on attempt {attempt + 1}/{_max_retries + 1}")
                            print(f"[RATE LIMIT] Waiting {wait_time:.1f}s before retry...")
                            print(f"[RATE LIMIT] Provider: {ModelConfig.LLM_PROVIDER}, Model: {ModelConfig.PRIMARY_MODEL}")
                            time.sleep(wait_time)
                        else:
                            print(f"\n[RATE LIMIT] Max retries ({_max_retries}) exceeded")
                            raise RateLimitError(
                                f"Rate limit exceeded after {_max_retries} retries. "
                                f"Last error: {str(e)}"
                            ) from e
                    else:
                        # Non-rate-limit error, re-raise immediately
                        raise
            
            # Should not reach here, but just in case
            if last_error:
                raise last_error
                
        return wrapper
    return decorator


def rate_limited_call(
    func: Callable,
    *args,
    max_retries: Optional[int] = None,
    retry_delay: Optional[float] = None,
    delay_between_calls: Optional[float] = None,
    **kwargs
) -> Any:
    """
    Execute a function with rate limit retry logic.
    
    Args:
        func: Function to call
        *args: Arguments to pass to the function
        max_retries: Maximum number of retries
        retry_delay: Delay in seconds between retries
        delay_between_calls: Delay after successful call
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        Result of the function call
    
    Usage:
        result = rate_limited_call(llm.generate, "prompt", max_retries=3)
    """
    _max_retries = max_retries if max_retries is not None else EvalConfig.RATE_LIMIT_MAX_RETRIES
    _retry_delay = retry_delay if retry_delay is not None else EvalConfig.RATE_LIMIT_RETRY_DELAY
    _delay_between = delay_between_calls if delay_between_calls is not None else EvalConfig.RATE_LIMIT_DELAY
    
    last_error = None
    
    for attempt in range(_max_retries + 1):
        try:
            result = func(*args, **kwargs)
            
            # Add delay after successful call
            if _delay_between > 0:
                time.sleep(_delay_between)
            
            return result
            
        except Exception as e:
            last_error = e
            
            if is_rate_limit_error(e):
                if attempt < _max_retries:
                    wait_time = _retry_delay * (1.5 ** attempt)
                    print(f"\n[RATE LIMIT] Hit rate limit on attempt {attempt + 1}/{_max_retries + 1}")
                    print(f"[RATE LIMIT] Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"\n[RATE LIMIT] Max retries ({_max_retries}) exceeded")
                    raise RateLimitError(
                        f"Rate limit exceeded after {_max_retries} retries. "
                        f"Last error: {str(e)}"
                    ) from e
            else:
                raise
    
    if last_error:
        raise last_error


class RateLimitedEvaluator:
    """
    A wrapper class for running evaluations with rate limit protection.
    Tracks API calls and automatically manages delays.
    """
    
    def __init__(
        self,
        delay_between_calls: float = None,
        retry_delay: float = None,
        max_retries: int = None
    ):
        self.delay_between_calls = delay_between_calls or EvalConfig.RATE_LIMIT_DELAY
        self.retry_delay = retry_delay or EvalConfig.RATE_LIMIT_RETRY_DELAY
        self.max_retries = max_retries or EvalConfig.RATE_LIMIT_MAX_RETRIES
        self.call_count = 0
        self.rate_limit_hits = 0
        self.start_time = None
    
    def start_evaluation(self):
        """Mark the start of an evaluation run."""
        self.call_count = 0
        self.rate_limit_hits = 0
        self.start_time = time.time()
        print(f"\n[EVAL] Starting evaluation with rate limit protection")
        print(f"[EVAL] Delay between calls: {self.delay_between_calls}s")
        print(f"[EVAL] Retry delay on rate limit: {self.retry_delay}s")
        print(f"[EVAL] Max retries: {self.max_retries}")
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with rate limit protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments for the function
        
        Returns:
            Function result
        """
        self.call_count += 1
        
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                
                # Delay after successful call
                time.sleep(self.delay_between_calls)
                
                return result
                
            except Exception as e:
                if is_rate_limit_error(e):
                    self.rate_limit_hits += 1
                    
                    if attempt < self.max_retries:
                        wait_time = self.retry_delay * (1.5 ** attempt)
                        print(f"\n[RATE LIMIT] Hit #{self.rate_limit_hits} - Attempt {attempt + 1}/{self.max_retries + 1}")
                        print(f"[RATE LIMIT] Waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        raise RateLimitError(
                            f"Rate limit exceeded after {self.max_retries} retries"
                        ) from e
                else:
                    raise
        
        return None
    
    def get_stats(self) -> dict:
        """Get evaluation statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            "total_calls": self.call_count,
            "rate_limit_hits": self.rate_limit_hits,
            "elapsed_time": elapsed,
            "avg_time_per_call": elapsed / self.call_count if self.call_count > 0 else 0
        }
    
    def print_stats(self):
        """Print evaluation statistics."""
        stats = self.get_stats()
        print(f"\n[EVAL STATS]")
        print(f"  Total API calls: {stats['total_calls']}")
        print(f"  Rate limit hits: {stats['rate_limit_hits']}")
        print(f"  Total time: {stats['elapsed_time']:.1f}s")
        print(f"  Avg time per call: {stats['avg_time_per_call']:.2f}s")
