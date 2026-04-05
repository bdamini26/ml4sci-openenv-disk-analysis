"""
Disk Image Analysis Environment Models
Pydantic models for observations, actions, and rewards
"""

import pydantic
from typing import List, Optional
import numpy as np


class Observation(pydantic.BaseModel):
    """What the agent sees at each step"""
    
    disk_image: List[List[float]] = pydantic.Field(
        ...,
        description="32x32 normalized grayscale disk image"
    )
    
    task_id: int = pydantic.Field(
        ...,
        description="Current task ID (1=detection, 2=classification, 3=habitability)"
    )
    
    step_count: int = pydantic.Field(
        ...,
        description="Number of steps taken so far"
    )
    
    last_action_feedback: Optional[str] = pydantic.Field(
        default=None,
        description="Feedback on previous action"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "disk_image": [[0.0, 0.1, ...], ...],
                "task_id": 1,
                "step_count": 0,
                "last_action_feedback": None
            }
        }


class Action(pydantic.BaseModel):
    """What the agent can do"""
    
    analysis_type: str = pydantic.Field(
        ...,
        description="Type of analysis: 'detect_transit', 'classify_disk', 'estimate_properties'"
    )
    
    confidence: float = pydantic.Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in its prediction (0.0-1.0)"
    )
    
    prediction: Optional[str] = pydantic.Field(
        default=None,
        description="The actual prediction/answer (e.g., 'yes_transit', 'young_disk', '0.65')"
    )
    
    reasoning: Optional[str] = pydantic.Field(
        default=None,
        description="Agent's reasoning for the prediction"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "analysis_type": "detect_transit",
                "confidence": 0.85,
                "prediction": "yes_transit",
                "reasoning": "Detected dip in brightness at expected time"
            }
        }


class Reward(pydantic.BaseModel):
    """Reward signal for the agent"""
    
    task_reward: float = pydantic.Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Reward for task completion (0.0-1.0)"
    )
    
    accuracy_bonus: float = pydantic.Field(
        ...,
        ge=-0.1,
        le=0.1,
        description="Bonus/penalty for accuracy (-0.1 to +0.1)"
    )
    
    confidence_penalty: float = pydantic.Field(
        ...,
        ge=-0.1,
        le=0.0,
        description="Penalty for overconfidence (-0.1 to 0.0)"
    )
    
    total_reward: float = pydantic.Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Sum of all rewards clamped to [0, 1]"
    )
    
    reward_summary: str = pydantic.Field(
        ...,
        description="Human-readable summary of reward"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_reward": 0.8,
                "accuracy_bonus": 0.1,
                "confidence_penalty": -0.05,
                "total_reward": 0.85,
                "reward_summary": "Good prediction with high confidence"
            }
        }