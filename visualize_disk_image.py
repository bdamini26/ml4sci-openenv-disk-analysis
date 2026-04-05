#!/usr/bin/env python3
"""
Visualize disk images from the environment API
"""

import asyncio
import numpy as np
from environment import DiskAnalysisEnv
from models import Action

async def visualize_disk_image():
    """Visualize a disk image as ASCII art"""
    
    env = DiskAnalysisEnv()
    
    # Reset to get image
    obs = await env.reset()
    
    # Get the disk image
    disk_image = np.array(obs.disk_image)
    
    print("\n" + "="*50)
    print("DISK IMAGE VISUALIZATION")
    print("="*50)
    print(f"Task: {obs.task_id}")
    print(f"Shape: {disk_image.shape}")
    print("\nASCII Visualization:")
    print("-" * 50)
    
    # Create ASCII art
    for row in disk_image:
        for val in row:
            if val > 0.8:
                print("█", end="")
            elif val > 0.6:
                print("▓", end="")
            elif val > 0.4:
                print("▒", end="")
            elif val > 0.2:
                print("░", end="")
            else:
                print(" ", end="")
        print()
    
    print("-" * 50)
    print("\nBrightness Scale:")
    print("  █ = Very bright (>0.8)")
    print("  ▓ = Bright (>0.6)")
    print("  ▒ = Medium (>0.4)")
    print("  ░ = Dim (>0.2)")
    print("    = Dark (≤0.2)")
    print("\n" + "="*50)
    
    # Show statistics
    print(f"\nImage Statistics:")
    print(f"  Min value: {disk_image.min():.3f}")
    print(f"  Max value: {disk_image.max():.3f}")
    print(f"  Mean value: {disk_image.mean():.3f}")
    print(f"  Pixels with value > 0.5: {(disk_image > 0.5).sum()} / {disk_image.size}")
    
    # Now test with action
    print("\n" + "="*50)
    print("TESTING ENVIRONMENT STEP")
    print("="*50)
    
    action = Action(
        analysis_type="detect_transit",
        confidence=0.85,
        prediction="yes_transit",
        reasoning="Testing visualization"
    )
    
    obs, reward, done, info = await env.step(action)
    
    print(f"\nReward: {reward.total_reward}")
    print(f"Task Reward: {reward.task_reward}")
    print(f"Accuracy Bonus: {reward.accuracy_bonus}")
    print(f"Summary: {reward.reward_summary}")
    print(f"Done: {done}")
    print(f"Ground Truth: {info['ground_truth']}")

if __name__ == "__main__":
    asyncio.run(visualize_disk_image())