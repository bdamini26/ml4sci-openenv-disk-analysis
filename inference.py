#!/usr/bin/env python3
"""
Inference Script: Baseline Agent for Disk Image Analysis Environment
Runs an OpenAI-powered agent against the environment and produces reproducible scores

Required environment variables:
- API_BASE_URL: LLM API endpoint
- MODEL_NAME: Model identifier
- HF_TOKEN: Hugging Face token
"""

import asyncio
import json
import os
import sys
from typing import List, Dict, Any
from datetime import datetime

from openai import OpenAI # type: ignore
import numpy as np

# Import environment
from models import Observation, Action, Reward
from environment import DiskAnalysisEnv
# ========== Configuration ==========

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN", "")

TEMPERATURE = 0.7
MAX_TOKENS = 500
MAX_STEPS = 20
MAX_TOTAL_REWARD = 3.0  # 3 tasks × max 1.0 reward
SUCCESS_SCORE_THRESHOLD = 0.65

TASK_NAME = "disk-image-analysis"
BENCHMARK = "ml4sci-disk-analysis"

# ========== Logging Functions ==========

def log_start(task: str, env: str, model: str) -> None:
    """Log start of evaluation"""
    timestamp = datetime.now().isoformat()
    print(f"[START] timestamp={timestamp} task={task} env={env} model={model}")
    sys.stdout.flush()


def log_step(step: int, action: str, reward: float, done: bool, error: Any = None) -> None:
    """Log each step"""
    log_line = f"[STEP] step={step} reward={reward:.2f} done={done}"
    if error:
        log_line += f" error={error}"
    print(log_line)
    sys.stdout.flush()


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log end of evaluation"""
    total_reward = sum(rewards)
    print(f"[END] success={success} steps={steps} score={score:.4f} total_reward={total_reward:.4f}")
    sys.stdout.flush()


# ========== Agent Functions ==========

def format_image_for_llm(image: List[List[float]]) -> str:
    """Convert image matrix to text description for LLM"""
    arr = np.array(image)
    # Create a simple ASCII representation
    ascii_image = ""
    for row in arr:
        for val in row:
            if val > 0.8:
                ascii_image += "█"
            elif val > 0.6:
                ascii_image += "▓"
            elif val > 0.4:
                ascii_image += "▒"
            elif val > 0.2:
                ascii_image += "░"
            else:
                ascii_image += " "
        ascii_image += "\n"
    
    return ascii_image


def get_model_message(
    client: OpenAI,
    step: int,
    observation: Observation,
    last_reward: float,
    history: List[str]
) -> Action:
    """
    Get agent's action from LLM model
    """
    
    task_descriptions = {
        1: "Transit Detection (Binary): Analyze disk image and determine if a planet transit occurred.",
        2: "Disk Classification (3-class): Classify the disk type as young_disk, evolved_disk, or debris_disk.",
        3: "Property Estimation (Regression): Estimate numeric disk properties (mass, inclination, scale height)."
    }
    
    task_desc = task_descriptions.get(observation.task_id, "Unknown task")
    image_visual = format_image_for_llm(observation.disk_image)
    
    system_prompt = f"""You are an expert astronomical image analysis agent trained in ML4SCI.
Your task: {task_desc}

You analyze protoplanetary disk images and make predictions about their properties.
For each analysis, provide:
1. Your prediction (specific format depends on task)
2. Your confidence level (0.0-1.0)
3. Brief reasoning

Task {observation.task_id} Format:
{get_task_format(observation.task_id)}

Previous reward: {last_reward:.2f}
Step: {step}/{MAX_STEPS}

Respond in JSON format ONLY:
{{
  "analysis_type": "detect_transit|classify_disk|estimate_properties",
  "confidence": 0.0-1.0,
  "prediction": "your_prediction",
  "reasoning": "brief explanation"
}}"""
    
    user_prompt = f"""Analyze this disk image (ASCII visualization):

{image_visual}

Make your prediction for task {observation.task_id}: {task_desc}

Respond with JSON only, no markdown, no other text."""
    
    try:
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        
        response_text = (completion.choices[0].message.content or "").strip()
        
        # Parse JSON response
        try:
            response_json = json.loads(response_text)
            action = Action(
                analysis_type=response_json.get("analysis_type", "detect_transit"),
                confidence=min(1.0, max(0.0, float(response_json.get("confidence", 0.5)))),
                prediction=response_json.get("prediction", "unknown"),
                reasoning=response_json.get("reasoning", "")
            )
            return action
        except (json.JSONDecodeError, ValueError):
            # Fallback action if parsing fails
            return Action(
                analysis_type="detect_transit",
                confidence=0.5,
                prediction="error_parsing",
                reasoning="Failed to parse model response"
            )
    
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return Action(
            analysis_type="detect_transit",
            confidence=0.5,
            prediction="error",
            reasoning=str(exc)
        )


def get_task_format(task_id: int) -> str:
    """Get expected format for task predictions"""
    if task_id == 1:
        return 'prediction: "yes_transit" or "no_transit"'
    elif task_id == 2:
        return 'prediction: "young_disk" or "evolved_disk" or "debris_disk"'
    else:
        return 'prediction: numeric value (e.g., "0.5" for mass in Jupiter masses)'


# ========== Main Evaluation Loop ==========

async def main() -> None:
    """Main evaluation function"""
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = DiskAnalysisEnv()
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # Reset environment
        result = await env.reset()
        last_reward = 0.0
        
        for step in range(1, MAX_STEPS + 1):
            if env.done:
                break
            
            # Get agent's action
            action = get_model_message(client, step, result, last_reward, history)
            
            # Execute step
            obs, reward, done, info = await env.step(action)
            
            reward_value = reward.total_reward
            done = done or step >= MAX_STEPS
            
            rewards.append(reward_value)
            steps_taken = step
            last_reward = reward_value
            
            # Log step
            log_step(step=step, action=action.prediction or "none", reward=reward_value, done=done)
            
            history.append(f"Step {step}: {action.prediction} -> reward {reward_value:+.2f}")
            
            result = obs
            
            if done:
                break
        
        # Calculate final score
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD
    
    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)
        success = False
        score = 0.0
    
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())