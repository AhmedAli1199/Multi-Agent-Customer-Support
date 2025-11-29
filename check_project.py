"""
Project status checker - verifies all components are in place
"""
from pathlib import Path
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    if Path(filepath).exists():
        print(f"  [OK] {description}")
        return True
    else:
        print(f"  [MISSING] {description}")
        return False

def main():
    """Run project status check"""
    print("="*80)
    print("MULTI-AGENT CUSTOMER SUPPORT SYSTEM - PROJECT STATUS CHECK")
    print("="*80)

    project_root = Path(__file__).parent
    checks_passed = 0
    total_checks = 0

    # Core source files
    print("\n[1] CORE SOURCE FILES")
    print("-"*80)
    files = [
        ("src/config.py", "Configuration module"),
        ("src/agents/base_agent.py", "Base agent class"),
        ("src/agents/triage_agent.py", "Triage agent"),
        ("src/agents/knowledge_agent.py", "Knowledge agent"),
        ("src/agents/action_agent.py", "Action agent"),
        ("src/agents/followup_agent.py", "Follow-up agent"),
        ("src/agents/escalation_agent.py", "Escalation agent"),
        ("src/baseline/single_agent.py", "Single-agent baseline"),
        ("src/orchestration/state.py", "LangGraph state"),
        ("src/orchestration/graph.py", "LangGraph workflow"),
        ("src/tools/knowledge_retrieval.py", "Knowledge retrieval"),
        ("src/tools/mock_apis.py", "Mock APIs"),
        ("src/utils/helpers.py", "Helper utilities"),
    ]
    for filepath, desc in files:
        total_checks += 1
        if check_file_exists(project_root / filepath, desc):
            checks_passed += 1

    # API and evaluation
    print("\n[2] API AND EVALUATION")
    print("-"*80)
    files = [
        ("src/api/app.py", "FastAPI application"),
        ("src/evaluation/metrics.py", "Evaluation metrics"),
        ("src/evaluation/ablation_study.py", "Ablation study"),
    ]
    for filepath, desc in files:
        total_checks += 1
        if check_file_exists(project_root / filepath, desc):
            checks_passed += 1

    # Scripts
    print("\n[3] SETUP AND TEST SCRIPTS")
    print("-"*80)
    files = [
        ("scripts/prepare_dataset.py", "Dataset preparation"),
        ("scripts/setup_vector_store.py", "Vector store setup"),
        ("test_system.py", "Multi-agent test"),
        ("test_baseline.py", "Baseline comparison test"),
        ("run_evaluation.py", "Full evaluation pipeline"),
    ]
    for filepath, desc in files:
        total_checks += 1
        if check_file_exists(project_root / filepath, desc):
            checks_passed += 1

    # Data files
    print("\n[4] DATA FILES")
    print("-"*80)
    files = [
        ("data/knowledge_base.json", "Knowledge base (126 entries)"),
        ("data/test_conversations.json", "Test dataset (100 conversations)"),
        ("data/bitext_dataset.json", "Full Bitext dataset"),
    ]
    for filepath, desc in files:
        total_checks += 1
        if check_file_exists(project_root / filepath, desc):
            checks_passed += 1

    # Check Chroma directory
    total_checks += 1
    if (project_root / "data" / "chroma_db").exists():
        print(f"  [OK] Chroma vector store directory")
        checks_passed += 1
    else:
        print(f"  [MISSING] Chroma vector store (run setup_vector_store.py)")

    # Docker files
    print("\n[5] DOCKER CONFIGURATION")
    print("-"*80)
    files = [
        ("Dockerfile", "Docker image definition"),
        ("docker-compose.yml", "Docker compose configuration"),
        (".dockerignore", "Docker ignore file"),
    ]
    for filepath, desc in files:
        total_checks += 1
        if check_file_exists(project_root / filepath, desc):
            checks_passed += 1

    # Documentation
    print("\n[6] DOCUMENTATION")
    print("-"*80)
    files = [
        ("README.md", "Main documentation"),
        ("CLAUDE.md", "Claude guidance file"),
        ("PROJECT_SUMMARY.md", "Project summary"),
        ("prompts.txt", "Prompts documentation"),
        (".env", "Environment configuration"),
        ("pyproject.toml", "UV dependencies"),
    ]
    for filepath, desc in files:
        total_checks += 1
        if check_file_exists(project_root / filepath, desc):
            checks_passed += 1

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Checks Passed: {checks_passed}/{total_checks}")
    print(f"Completion: {(checks_passed/total_checks)*100:.1f}%")

    if checks_passed == total_checks:
        print("\n[SUCCESS] All components in place!")
        print("\nNext steps:")
        print("  1. Test the system: .venv/Scripts/python.exe test_system.py")
        print("  2. Run evaluation: .venv/Scripts/python.exe run_evaluation.py")
        print("  3. Write research paper using results")
        print("  4. Package for submission")
    else:
        print(f"\n[WARNING] {total_checks - checks_passed} components missing")
        print("Review missing files above and complete setup")

    print("="*80)

    # Additional checks
    print("\n[ADDITIONAL INFORMATION]")
    print("-"*80)

    # Check if vector store is populated
    chroma_dir = project_root / "data" / "chroma_db"
    if chroma_dir.exists():
        files_count = len(list(chroma_dir.rglob("*")))
        print(f"  - Chroma DB files: {files_count}")

    # Check knowledge base size
    kb_file = project_root / "data" / "knowledge_base.json"
    if kb_file.exists():
        import json
        with open(kb_file, 'r', encoding='utf-8') as f:
            kb = json.load(f)
        print(f"  - Knowledge base entries: {len(kb)}")

    # Check test dataset size
    test_file = project_root / "data" / "test_conversations.json"
    if test_file.exists():
        import json
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"  - Test conversations: {len(test_data)}")

    # Check if .env has API key
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            env_content = f.read()
        if "GEMINI_API_KEY=AIzaSy" in env_content:
            print(f"  - Gemini API key: SET")
        else:
            print(f"  - Gemini API key: NOT SET (add to .env file)")

    print("="*80)

if __name__ == "__main__":
    main()
