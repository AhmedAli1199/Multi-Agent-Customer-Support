"""
Helper utilities for the multi-agent customer support system.
"""
import time
import json
from typing import Dict, List, Any
from datetime import datetime

def format_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()

def measure_latency(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        latency = end_time - start_time
        return result, latency
    return wrapper

def extract_order_id(text: str) -> str:
    """Extract order ID from text (simple pattern matching)"""
    import re
    # Look for patterns like: #12345, Order #12345, order 12345
    patterns = [
        r'#(\d{5,})',
        r'[Oo]rder\s*#?(\d{5,})',
        r'ID\s*#?(\d{5,})'
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    return None

def calculate_sentiment_score(text: str) -> float:
    """
    Simple sentiment analysis based on keywords.
    Returns score between -1 (very negative) and 1 (very positive)
    """
    negative_keywords = [
        'angry', 'frustrated', 'terrible', 'awful', 'horrible', 'bad',
        'disappointed', 'upset', 'hate', 'worst', 'useless', 'broken',
        'damaged', 'never', 'unacceptable', 'ridiculous'
    ]

    positive_keywords = [
        'great', 'excellent', 'love', 'perfect', 'amazing', 'wonderful',
        'fantastic', 'good', 'thank', 'appreciate', 'satisfied', 'happy'
    ]

    text_lower = text.lower()

    negative_count = sum(1 for word in negative_keywords if word in text_lower)
    positive_count = sum(1 for word in positive_keywords if word in text_lower)

    total = negative_count + positive_count
    if total == 0:
        return 0.0  # Neutral

    score = (positive_count - negative_count) / total
    return score

def format_agent_message(agent_name: str, content: str, metadata: Dict = None) -> Dict:
    """Format a message from an agent"""
    message = {
        "agent": agent_name,
        "content": content,
        "timestamp": format_timestamp()
    }

    if metadata:
        message["metadata"] = metadata

    return message

def save_prompts_to_file(prompts: List[Dict[str, str]], filepath: str):
    """Save prompts to a text file for submission"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PROMPTS USED IN MULTI-AGENT CUSTOMER SUPPORT SYSTEM\n")
        f.write("="*80 + "\n\n")

        for i, prompt_entry in enumerate(prompts, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"PROMPT #{i}: {prompt_entry['name']}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Context: {prompt_entry.get('context', 'N/A')}\n\n")
            f.write(f"Prompt:\n{prompt_entry['prompt']}\n")

    print(f"[OK] Prompts saved to: {filepath}")
