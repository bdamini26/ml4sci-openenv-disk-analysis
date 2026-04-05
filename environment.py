"""
Disk Image Analysis Environment
Simulates real-world protoplanetary disk analysis tasks using machine learning
"""

import asyncio
import random
import numpy as np
from typing import Optional, Tuple, Dict, Any
from models import Observation, Action, Reward


class DiskAnalysisEnv:
    """
    OpenEnv environment for disk image analysis.
    
    Tasks:
    1. Transit Detection (Easy): Detect if a planet transit occurred in light curve
    2. Disk Classification (Medium): Classify disk type (young/evolved/debris)
    3. Property Estimation (Hard): Estimate disk properties (mass, age, inclination)
    """
    
    def __init__(self):
        self.task_id = 1
        self.step_count = 0
        self.max_steps = 20
        self.done = False
        self.episode_data = {}
        self.current_image = None
        self.ground_truth = None
        self.history = []
        
    async def reset(self) -> Observation:
        """Reset environment and start a new episode"""
        self.step_count = 0
        self.done = False
        self.history = []
        
        # Generate synthetic disk image (32x32)
        self.current_image = self._generate_disk_image()
        
        # Generate ground truth for current task
        self.ground_truth = self._generate_ground_truth()
        
        return Observation(
            disk_image=self.current_image.tolist(),
            task_id=self.task_id,
            step_count=0,
            last_action_feedback=None
        )
    
    async def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        self.step_count += 1
        
        # Evaluate the action
        reward = self._evaluate_action(action)
        
        # Episode ends after prediction or max steps
        done = self.step_count >= self.max_steps or action.prediction is not None
        
        if done:
            # Move to next task for multi-task learning
            if self.task_id < 3:
                self.task_id += 1
            else:
                self.task_id = 1  # Reset to first task
        
        # Create next observation
        next_observation = Observation(
            disk_image=self.current_image.tolist(),
            task_id=self.task_id,
            step_count=self.step_count,
            last_action_feedback=f"Reward: {reward.total_reward:.2f}"
        )
        
        info = {
            "ground_truth": self.ground_truth,
            "task_name": self._get_task_name(),
            "reward_breakdown": {
                "task_reward": reward.task_reward,
                "accuracy_bonus": reward.accuracy_bonus,
                "confidence_penalty": reward.confidence_penalty
            }
        }
        
        return next_observation, reward, done, info
    
    async def state(self) -> Dict[str, Any]:
        """Return current environment state"""
        return {
            "task_id": self.task_id,
            "step_count": self.step_count,
            "done": self.done,
            "ground_truth": self.ground_truth,
            "current_image_shape": self.current_image.shape if self.current_image is not None else None,
            "history": self.history
        }
    
    # ========== Private Methods ==========
    
    def _generate_disk_image(self) -> np.ndarray:
        """Generate synthetic disk image (32x32 normalized grayscale)"""
        image = np.zeros((32, 32))
        
        # Create a disk-like structure
        center_x, center_y = 16, 16
        for x in range(32):
            for y in range(32):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < 14:
                    # Disk brightness inversely related to distance from center
                    brightness = max(0, 1 - (dist / 14))
                    image[x, y] = brightness + np.random.normal(0, 0.05)
        
        # Normalize to [0, 1]
        image = np.clip(image, 0, 1)
        
        return image
    
    def _generate_ground_truth(self) -> Dict[str, Any]:
        """Generate ground truth for current task"""
        if self.task_id == 1:  # Transit detection
            return {
                "has_transit": random.choice([True, False]),
                "transit_depth": random.uniform(0.01, 0.1) if random.random() > 0.5 else 0,
                "transit_time": random.uniform(0.3, 0.7) if random.random() > 0.5 else None
            }
        
        elif self.task_id == 2:  # Disk classification
            disk_types = ["young_disk", "evolved_disk", "debris_disk"]
            return {
                "disk_type": random.choice(disk_types),
                "age_gyr": random.uniform(0.1, 10.0),
                "dust_continuum_detected": random.choice([True, False])
            }
        
        else:  # Task 3: Property estimation
            return {
                "disk_mass_mjup": random.uniform(0.001, 1.0),
                "inclination_deg": random.uniform(0, 90),
                "scale_height_au": random.uniform(0.01, 0.5),
                "is_habitable": random.choice([True, False])
            }
    
    def _evaluate_action(self, action: Action) -> Reward:
        """Evaluate agent's action and return reward"""
        
        if self.task_id == 1:
            reward = self._evaluate_transit_detection(action)
        elif self.task_id == 2:
            reward = self._evaluate_disk_classification(action)
        else:
            reward = self._evaluate_property_estimation(action)
        
        return reward
    
    def _evaluate_transit_detection(self, action: Action) -> Reward:
        """Evaluate transit detection task"""
        task_reward = 0.0
        accuracy_bonus = 0.0
        confidence_penalty = 0.0
        
        if action.prediction and self.ground_truth:
            # Check if prediction is correct
            agent_detected = action.prediction.lower() == "yes_transit"
            ground_truth_transit = self.ground_truth.get("has_transit", False)
            
            if agent_detected == ground_truth_transit:
                task_reward = 0.8
                accuracy_bonus = 0.1 if action.confidence > 0.8 else 0.05
            else:
                task_reward = 0.2
                accuracy_bonus = -0.1
            
            # Penalize overconfidence on wrong predictions
            if agent_detected != ground_truth_transit and action.confidence > 0.9:
                confidence_penalty = -0.1
            elif action.confidence > 0.95:
                confidence_penalty = -0.05
        
        total_reward = np.clip(task_reward + accuracy_bonus + confidence_penalty, 0, 1)
        
        return Reward(
            task_reward=max(0, task_reward),
            accuracy_bonus=accuracy_bonus,
            confidence_penalty=confidence_penalty,
            total_reward=float(total_reward),
            reward_summary=f"Transit detection: {'Correct' if task_reward > 0.5 else 'Incorrect'}"
        )
    
    def _evaluate_disk_classification(self, action: Action) -> Reward:
        """Evaluate disk classification task"""
        task_reward = 0.0
        accuracy_bonus = 0.0
        confidence_penalty = 0.0
        
        if action.prediction and self.ground_truth:
            ground_truth_type = self.ground_truth.get("disk_type", "young_disk")
            agent_prediction = action.prediction.lower()
            
            if agent_prediction == ground_truth_type:
                task_reward = 0.8
                accuracy_bonus = 0.1 if action.confidence > 0.8 else 0.05
            else:
                # Partial credit for reasonable misclassification
                task_reward = 0.3
                accuracy_bonus = -0.05
            
            # Confidence penalty
            if action.confidence > 0.95:
                confidence_penalty = -0.05
        
        total_reward = np.clip(task_reward + accuracy_bonus + confidence_penalty, 0, 1)
        
        return Reward(
            task_reward=max(0, task_reward),
            accuracy_bonus=accuracy_bonus,
            confidence_penalty=confidence_penalty,
            total_reward=float(total_reward),
            reward_summary=f"Classification: {'Correct' if task_reward > 0.5 else 'Incorrect'}"
        )
    
    def _evaluate_property_estimation(self, action: Action) -> Reward:
        """Evaluate property estimation task"""
        task_reward = 0.0
        accuracy_bonus = 0.0
        confidence_penalty = 0.0
        
        if action.prediction and self.ground_truth:
            try:
                # Try to parse prediction as numeric
                agent_value = float(action.prediction)
                
                # Get ground truth property
                if "mass" in action.analysis_type.lower():
                    ground_truth_value = self.ground_truth.get("disk_mass_mjup", 0.1)
                    error = abs(agent_value - ground_truth_value) / ground_truth_value
                elif "inclination" in action.analysis_type.lower():
                    ground_truth_value = self.ground_truth.get("inclination_deg", 45)
                    error = abs(agent_value - ground_truth_value) / 90.0
                else:
                    ground_truth_value = self.ground_truth.get("scale_height_au", 0.1)
                    error = abs(agent_value - ground_truth_value) / ground_truth_value
                
                # Score based on error
                if error < 0.1:
                    task_reward = 0.9
                    accuracy_bonus = 0.05
                elif error < 0.3:
                    task_reward = 0.7
                    accuracy_bonus = 0.0
                elif error < 0.5:
                    task_reward = 0.4
                    accuracy_bonus = -0.05
                else:
                    task_reward = 0.1
                    accuracy_bonus = -0.1
                
            except (ValueError, TypeError):
                task_reward = 0.2
                accuracy_bonus = -0.1
            
            # Confidence penalty
            if action.confidence > 0.95:
                confidence_penalty = -0.05
        
        total_reward = np.clip(task_reward + accuracy_bonus + confidence_penalty, 0, 1)
        
        return Reward(
            task_reward=max(0, task_reward),
            accuracy_bonus=accuracy_bonus,
            confidence_penalty=confidence_penalty,
            total_reward=float(total_reward),
            reward_summary=f"Estimation: {'Good' if task_reward > 0.6 else 'Poor'}"
        )
    
    def _get_task_name(self) -> str:
        """Get human-readable task name"""
        tasks = {
            1: "Transit Detection",
            2: "Disk Classification",
            3: "Property Estimation"
        }
        return tasks.get(self.task_id, "Unknown")