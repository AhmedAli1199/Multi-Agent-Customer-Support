"""
FastAPI application for Multi-Agent Customer Support System
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.graph_v2 import multi_agent_workflow_v2
from baseline.single_agent import single_agent
from config import PROJECT_ROOT
import time

# Create FastAPI app
app = FastAPI(
    title="Multi-Agent Customer Support System",
    description="AI-powered customer support using collaborative multi-agent architecture",
    version="1.0.0"
)

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    conversation_history: Optional[List[Dict]] = []
    use_multi_agent: bool = True  # Switch between multi-agent and single-agent
    auto_execute: bool = False  # Auto-execute actions for demo

class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    intent: Optional[str] = None
    agent_sequence: Optional[List[str]] = None
    resolution_status: Optional[str] = None
    needs_escalation: bool = False
    confidence_score: Optional[float] = None
    processing_time: float
    system_type: str  # "multi-agent" or "single-agent"

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    components: Dict[str, str]

@app.get("/")
async def root():
    """Serve the web interface"""
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/api/info", response_model=Dict[str, str])
async def api_info():
    """API information endpoint"""
    return {
        "message": "Multi-Agent Customer Support API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "components": {
            "multi_agent_system": "operational (V2)",
            "single_agent_baseline": "operational",
            "knowledge_base": "operational",
            "vector_store": "operational (with fallback)",
            "mock_apis": "operational"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process customer query using multi-agent or single-agent system

    Args:
        request: ChatRequest with message and optional conversation history

    Returns:
        ChatResponse with agent response and metadata
    """
    try:
        start_time = time.time()

        if request.use_multi_agent:
            # Use multi-agent system
            initial_state = {
                "customer_query": request.message,
                "conversation_history": request.conversation_history or [],
                "current_agent": None,
                "next_agent": None,
                "agent_sequence": [],
                "needs_escalation": False,
                "resolution_status": "unresolved",
                "triage_result": None,
                "knowledge_result": None,
                "action_result": None,
                "followup_result": None,
                "escalation_result": None,
                "final_response": None,
                "intent": None,
                "entities": None,
                "urgency": None,
                "sentiment": None,
                "confidence_score": None,
                "metadata": None
            }

            final_state = multi_agent_workflow_v2.invoke(initial_state)

            response = ChatResponse(
                response=final_state.get("final_response", "I apologize, but I couldn't process your request."),
                intent=final_state.get("intent"),
                agent_sequence=final_state.get("agent_sequence", []),
                resolution_status=final_state.get("resolution_status"),
                needs_escalation=final_state.get("needs_escalation", False),
                confidence_score=final_state.get("confidence_score"),
                processing_time=time.time() - start_time,
                system_type="multi-agent"
            )
        else:
            # Use single-agent baseline
            result = single_agent.process(
                request.message,
                conversation_history=request.conversation_history,
                auto_execute=request.auto_execute
            )

            response = ChatResponse(
                response=result.get("response", "I apologize, but I couldn't process your request."),
                intent=result.get("intent"),
                agent_sequence=["single-agent"],
                resolution_status="resolved" if not result.get("needs_escalation") else "escalated",
                needs_escalation=result.get("needs_escalation", False),
                confidence_score=result.get("confidence"),
                processing_time=time.time() - start_time,
                system_type="single-agent"
            )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get system metrics (placeholder for evaluation metrics)"""
    return {
        "message": "Metrics endpoint - implement with evaluation results",
        "available_metrics": [
            "First-Contact Resolution (FCR)",
            "Average Response Time (ART)",
            "Escalation Rate (ER)",
            "Customer Satisfaction (CSAT)",
            "Intent Classification Accuracy"
        ]
    }

@app.get("/agents")
async def list_agents():
    """List available agents in the multi-agent system"""
    return {
        "multi_agent_system": {
            "agents": [
                {
                    "name": "Triage Agent",
                    "role": "Intent classification and routing",
                    "model": "gemini-2.5-pro"
                },
                {
                    "name": "Knowledge Agent",
                    "role": "FAQ handling with RAG",
                    "model": "gemini-2.5-flash"
                },
                {
                    "name": "Action Agent",
                    "role": "Backend operations execution",
                    "model": "gemini-2.5-pro"
                },
                {
                    "name": "Follow-Up Agent",
                    "role": "Customer satisfaction checks",
                    "model": "gemini-2.5-flash"
                },
                {
                    "name": "Escalation Agent",
                    "role": "Human handoff preparation",
                    "model": "gemini-2.5-pro"
                }
            ],
            "orchestration": "LangGraph StateGraph",
            "routing_logic": "Dynamic based on triage results"
        },
        "baseline_system": {
            "agent": "Single Agent",
            "role": "All-in-one customer support",
            "model": "gemini-2.5-pro"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
