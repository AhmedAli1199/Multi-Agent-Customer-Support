"""
Script to download and prepare the Bitext customer support dataset.
Creates both the main dataset and knowledge base for RAG.
"""
import json
import sys
from pathlib import Path
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset
from config import DATA_DIR, DATASET_FILE, KNOWLEDGE_BASE_FILE, TEST_DATASET_FILE, DatasetConfig

def download_bitext_dataset():
    """Download Bitext customer support dataset from Hugging Face"""
    print("[INFO] Downloading Bitext customer support dataset...")
    try:
        dataset = load_dataset(DatasetConfig.BITEXT_DATASET_NAME)
        print(f"[OK] Dataset downloaded: {len(dataset['train'])} examples")
        return dataset['train']
    except Exception as e:
        print(f"[ERROR] Error downloading dataset: {e}")
        print("[INFO] Creating synthetic sample dataset instead...")
        return create_sample_dataset()

def create_sample_dataset():
    """Create a sample dataset for testing if download fails"""
    sample_data = [
        {
            "instruction": "I want to cancel my order",
            "category": "cancel_order",
            "intent": "cancel_order",
            "response": "I understand you want to cancel your order. I can help you with that. May I have your order number?"
        },
        {
            "instruction": "Where is my refund?",
            "category": "check_refund_status",
            "intent": "track_refund",
            "response": "I'll help you track your refund. Could you please provide your order number?"
        },
        {
            "instruction": "How do I change my shipping address?",
            "category": "change_shipping_address",
            "intent": "edit_account",
            "response": "I can assist you with changing your shipping address. Please provide your account email."
        },
        {
            "instruction": "What is your return policy?",
            "category": "check_return_policy",
            "intent": "contact_customer_service",
            "response": "Our return policy allows returns within 30 days of purchase. Items must be in original condition with tags attached."
        },
        {
            "instruction": "I received a damaged product",
            "category": "complaint",
            "intent": "complaint",
            "response": "I'm sorry to hear your product arrived damaged. We'll make this right. Can you provide your order number and describe the damage?"
        }
    ]
    return sample_data

def prepare_knowledge_base(dataset, size=200):
    """Create knowledge base from dataset for RAG"""
    print(f"\n[INFO] Preparing knowledge base ({size} entries)...")

    knowledge_base = []
    seen_intents = set()

    for item in dataset:
        if len(knowledge_base) >= size:
            break

        intent = item.get("intent", item.get("category", "unknown"))

        # Try to get diverse intents
        if intent not in seen_intents or len(knowledge_base) < size // 2:
            kb_entry = {
                "id": len(knowledge_base),
                "question": item["instruction"],
                "answer": item["response"],
                "intent": intent,
                "category": item.get("category", intent)
            }
            knowledge_base.append(kb_entry)
            seen_intents.add(intent)

    # Save knowledge base
    with open(KNOWLEDGE_BASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)

    print(f"[OK] Knowledge base created: {len(knowledge_base)} entries")
    print(f"[OK] Saved to: {KNOWLEDGE_BASE_FILE}")

    return knowledge_base

def prepare_test_conversations(dataset, sample_size=100):
    """Prepare test conversations for evaluation"""
    print(f"\n[INFO] Preparing test dataset ({sample_size} conversations)...")

    # Sample from dataset
    if len(dataset) > sample_size:
        test_data = random.sample(list(dataset), sample_size)
    else:
        test_data = list(dataset)

    test_conversations = []
    for i, item in enumerate(test_data):
        conversation = {
            "conversation_id": f"test_{i:04d}",
            "customer_query": item["instruction"],
            "intent": item.get("intent", item.get("category", "unknown")),
            "expected_response_type": item.get("category", "info_query"),
            "ground_truth_action": None,  # Will be populated during evaluation
        }
        test_conversations.append(conversation)

    # Save test dataset
    with open(TEST_DATASET_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_conversations, f, indent=2, ensure_ascii=False)

    print(f"[OK] Test dataset created: {len(test_conversations)} conversations")
    print(f"[OK] Saved to: {TEST_DATASET_FILE}")

    return test_conversations

def prepare_full_dataset(dataset):
    """Save full dataset for reference"""
    print(f"\n[INFO] Saving full dataset...")

    full_data = []
    for item in dataset:
        full_data.append({
            "instruction": item["instruction"],
            "response": item["response"],
            "intent": item.get("intent", item.get("category", "unknown")),
            "category": item.get("category", "unknown")
        })

    with open(DATASET_FILE, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Full dataset saved: {len(full_data)} entries")
    print(f"[OK] Saved to: {DATASET_FILE}")

def main():
    print("="*60)
    print("Bitext Dataset Preparation")
    print("="*60)

    # Download dataset
    dataset = download_bitext_dataset()

    # Prepare knowledge base for RAG
    knowledge_base = prepare_knowledge_base(dataset, DatasetConfig.KNOWLEDGE_BASE_SIZE)

    # Prepare test conversations
    test_conversations = prepare_test_conversations(dataset, 100)

    # Save full dataset
    prepare_full_dataset(dataset)

    print("\n" + "="*60)
    print("[SUCCESS] Dataset preparation complete!")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  1. Knowledge base: {KNOWLEDGE_BASE_FILE}")
    print(f"  2. Test dataset: {TEST_DATASET_FILE}")
    print(f"  3. Full dataset: {DATASET_FILE}")
    print(f"\nNext steps:")
    print(f"  - Run: python scripts/setup_vector_store.py")
    print(f"  - This will create the Chroma vector store for RAG")

if __name__ == "__main__":
    main()
