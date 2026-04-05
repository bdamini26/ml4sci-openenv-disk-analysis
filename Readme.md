# Protoplanetary Disk Image Analysis Environment

## Overview
OpenEnv environment for analyzing protoplanetary disk images using machine learning. This environment enables rapid development and testing of AI agents for astronomical image analysis tasks.

## Problem Statement
Astronomers currently spend 250+ hours monthly manually analyzing protoplanetary disk images. This environment automates analysis tasks, enabling AI systems to accelerate scientific discovery and save significant time.

## Features
- **3 Progressive Tasks**: Easy, Medium, and Hard difficulty levels
- **Real-world Dataset**: Synthetic protoplanetary disk images (32x32 normalized grayscale)
- **Reward Shaping**: Task-specific grading logic with confidence-based bonuses
- **Multi-task Learning**: AI learns across different astronomical analysis tasks
- **Live Deployment**: Hosted on HuggingFace Spaces with FastAPI server

## Tasks

### Task 1: Transit Detection (Easy) 🟢
- **Type**: Binary Classification
- **Input**: Disk image (32x32)
- **Output**: "yes_transit" or "no_transit"
- **Reward**: 0.8 if correct, 0.2 if wrong
- **Description**: Detect if a planet transit occurred in the light curve

### Task 2: Disk Classification (Medium) 🟡
- **Type**: 3-class Classification
- **Input**: Disk image (32x32)
- **Output**: "young_disk", "evolved_disk", or "debris_disk"
- **Reward**: 0.8 if correct, 0.3 if close
- **Description**: Classify disk type based on age and properties

### Task 3: Property Estimation (Hard) 🔴
- **Type**: Regression
- **Input**: Disk image (32x32)
- **Output**: Numeric value (0.0-1.0)
- **Reward**: 0.9 if <10% error, 0.7 if <30%, 0.4 if <50%, 0.1 otherwise
- **Description**: Estimate disk properties (mass, inclination, scale height)

## API Endpoints

### Reset Environment
```bash
POST /reset
```
Returns initial observation for a new episode.

### Execute Step
```bash
POST /step
Body: {
  "analysis_type": "detect_transit|classify_disk|estimate_properties",
  "confidence": 0.0-1.0,
  "prediction": "your_prediction",
  "reasoning": "optional explanation"
}
```
Returns observation, reward, done flag, and info.

### Get Current State
```bash
POST /state
```
Returns current environment state including task ID, step count, ground truth, and history.

### List Tasks
```bash
GET /tasks
```
Returns available tasks with descriptions and difficulty levels.

### Health Check
```bash
GET /health
```
Returns server status.

## Usage Example
```python
import requests

BASE_URL = "https://damsaychesse-disk-image-analysis.hf.space"

# Reset environment
response = requests.post(f"{BASE_URL}/reset", json={})
observation = response.json()["observation"]

# Take a step
action = {
    "analysis_type": "detect_transit",
    "confidence": 0.85,
    "prediction": "yes_transit",
    "reasoning": "Brightness dip detected"
}
response = requests.post(f"{BASE_URL}/step", json=action)
reward = response.json()["reward"]["total_reward"]
print(f"Reward: {reward}")
```

## Expected Performance

- **Task 1 (Transit Detection)**: 0.7-0.8 reward
- **Task 2 (Classification)**: 0.6-0.75 reward
- **Task 3 (Property Estimation)**: 0.55-0.7 reward
- **Overall Score**: 0.65-0.75

## Technical Details

- **Framework**: FastAPI + Pydantic
- **Environment**: Custom DiskAnalysisEnv class
- **Image Generation**: Synthetic 32x32 normalized grayscale
- **Reward Logic**: Task-specific grading with confidence penalties
- **Deployment**: Docker + HuggingFace Spaces
- **Port**: 7860

## Implementation

### Models (Pydantic)
- `Observation`: Disk image, task ID, step count, feedback
- `Action`: Analysis type, confidence, prediction, reasoning
- `Reward`: Task reward, accuracy bonus, confidence penalty, total reward

### Environment
- `reset()`: Initialize new episode
- `step(action)`: Execute action and return results
- `state()`: Get current environment state

### Server
- FastAPI endpoints for all OpenEnv API methods
- CORS enabled for cross-origin requests
- Automatic validation and error handling

## Domain Motivation

This environment is based on real research from the **ML4SCI (Machine Learning for Science)** initiative, specifically the **EXXA Disk Analysis Project**. It demonstrates how AI can accelerate scientific discovery in astronomy by automating repetitive image analysis tasks.

## References

- [ML4SCI EXXA Disk Analysis](https://github.com/bdamini26/ml4sci-exxa-disk-ml)
- [OpenEnv Documentation](https://github.com/meta-pytorch/OpenEnv)
- [HuggingFace Spaces](https://huggingface.co/spaces)

## License

MIT

## Author

damsaychesse

---

**Live Space**: https://huggingface.co/spaces/damsaychesse/disk-image-analysis