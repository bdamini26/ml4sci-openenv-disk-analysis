"""
FastAPI server for Disk Image Analysis Environment
Exposes OpenEnv API endpoints: reset(), step(), state()
"""

import fastapi
from fastapi.responses import JSONResponse
import asyncio
import uvicorn
from typing import Dict, Any, Optional
from models import Observation, Action, Reward
from environment import DiskAnalysisEnv

# Initialize FastAPI app
app = fastapi.FastAPI(
    title="Disk Image Analysis Environment",
    description="OpenEnv environment for protoplanetary disk image analysis",
    version="1.0.0"
)

# Global environment instance
env: Optional[DiskAnalysisEnv] = None


@app.on_event("startup")
async def startup_event():
    """Initialize environment on startup"""
    global env
    env = DiskAnalysisEnv()
    print("✓ Environment initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("✓ Server shutting down")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "disk-image-analysis-env",
        "version": "1.0.0"
    }


@app.post("/reset")
async def reset_env() -> Dict[str, Any]:
    """
    Reset the environment and return initial observation
    
    OpenEnv API: reset() -> Observation
    """
    global env
    if env is None:
        env = DiskAnalysisEnv()
    
    try:
        obs = await env.reset()
        
        return {
            "status": "success",
            "observation": obs.model_dump(),
            "message": "Environment reset. Ready for new episode."
        }
    
    except Exception as e:
        raise fastapi.HTTPException(
            status_code=500,
            detail=f"Error resetting environment: {str(e)}"
        )


@app.post("/step")
async def step_env(action: Action) -> Dict[str, Any]:
    """
    Execute one step in the environment
    
    OpenEnv API: step(action) -> (Observation, Reward, Done, Info)
    """
    global env
    if env is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Environment not initialized. Call /reset first."}
        )
    
    try:
        obs, reward, done, info = await env.step(action)
        
        return {
            "status": "success",
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        }
    
    except Exception as e:
        raise fastapi.HTTPException(
            status_code=500,
            detail=f"Error executing step: {str(e)}"
        )


@app.post("/state")
async def get_state() -> Dict[str, Any]:
    """
    Get current environment state
    
    OpenEnv API: state() -> Dict
    """
    global env
    if env is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Environment not initialized. Call /reset first."}
        )
    
    try:
        state = await env.state()
        
        return {
            "status": "success",
            "state": state
        }
    
    except Exception as e:
        raise fastapi.HTTPException(
            status_code=500,
            detail=f"Error getting state: {str(e)}"
        )


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    """Get available tasks"""
    return {
        "tasks": [
            {
                "id": 1,
                "name": "Transit Detection",
                "difficulty": "easy",
                "description": "Binary classification: detect if transit occurred"
            },
            {
                "id": 2,
                "name": "Disk Classification",
                "difficulty": "medium",
                "description": "Multi-class classification: young/evolved/debris disk"
            },
            {
                "id": 3,
                "name": "Property Estimation",
                "difficulty": "hard",
                "description": "Regression: estimate mass, inclination, scale height"
            }
        ]
    }


@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "service": "Disk Image Analysis Environment",
        "version": "1.0.0",
        "openenv_api": {
            "reset": "POST /reset - Reset environment",
            "step": "POST /step - Execute one step",
            "state": "POST /state - Get current state",
            "tasks": "GET /tasks - List available tasks"
        },
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=7860,
        reload=False
    )