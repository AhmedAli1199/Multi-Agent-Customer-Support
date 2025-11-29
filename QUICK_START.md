# Quick Start Guide

## Testing the System (Fastest Options)

### 1. Quick Demo Test (30 seconds)
```bash
.venv/Scripts/python.exe test_system.py
```
Tests a single query through the multi-agent system.

### 2. Baseline Comparison (1 minute)
```bash
.venv/Scripts/python.exe test_baseline.py
```
Compares multi-agent vs single-agent on one query.

### 3. Full Evaluation (10-15 minutes)
```bash
.venv/Scripts/python.exe run_evaluation.py
```
Runs 20-query evaluation + ablation study. Results saved to `data/`.

## Running the API Server

```bash
# Start server
.venv/Scripts/python.exe -m uvicorn src.api.app:app --reload

# Access docs
http://localhost:8000/docs
```

## Testing via API

```bash
# Using curl
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to cancel my order #12345", "use_multi_agent": true}'

# Or use Swagger UI at http://localhost:8000/docs
```

##Common Issues

**Issue**: `ImportError: cannot import name 'triage_agent'`
**Fix**: Already fixed in the code - agents are now instantiated as classes

**Issue**: `Collection not found, using keyword-based fallback`
**Note**: This is expected - the system falls back to keyword search (still works!)

**Issue**: API key error
**Fix**: Ensure `.env` has `GEMINI_API_KEY=your_key_here`

## Project Status Check

```bash
.venv/Scripts/python.exe check_project.py
```

Shows which components are in place (should be 97-100% complete).

## For Your Paper

1. Run evaluation to get metrics:
   ```bash
   .venv/Scripts/python.exe run_evaluation.py
   ```

2. Check results in:
   - `data/comparison_results.json` - Multi-agent vs single-agent
   - `data/ablation_results.json` - Component analysis

3. Include these metrics in your research paper

## File Locations

- **Agents**: `src/agents/*.py`
- **Orchestration**: `src/orchestration/graph.py`
- **API**: `src/api/app.py`
- **Evaluation**: `src/evaluation/*.py`
- **Prompts**: `prompts.txt`
- **Configuration**: `src/config.py`

## Next Steps for Submission

1. ✅ Code implementation (DONE)
2. ⏳ Run evaluation (IN PROGRESS)
3. ⏳ Write research paper (use PROJECT_SUMMARY.md as guide)
4. ⏳ Package submission ZIP

Deadline: December 7, 2025
