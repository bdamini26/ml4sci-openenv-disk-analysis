import requests

BASE_URL = "https://damsaychesse-disk-image-analysis.hf.space"

print("=" * 50)
print("TESTING DISK IMAGE ANALYSIS ENVIRONMENT")
print("=" * 50)

# Step 1: Reset
print("\n1. Resetting environment...")
response = requests.post(f"{BASE_URL}/reset", json={})
print(f"   Status: {response.json()['status']}")
print(f"   Task ID: {response.json()['observation']['task_id']}")

# Step 2: Make Action
print("\n2. Taking a step...")
action = {
    "analysis_type": "detect_transit",
    "confidence": 0.85,
    "prediction": "yes_transit",
    "reasoning": "test"
}
response = requests.post(f"{BASE_URL}/step", json=action)
data = response.json()

print(f"   Status: {data['status']}")
print(f"   Task Reward: {data['reward']['task_reward']}")
print(f"   Accuracy Bonus: {data['reward']['accuracy_bonus']}")
print(f"   Total Reward: {data['reward']['total_reward']}")
print(f"   Summary: {data['reward']['reward_summary']}")

print("\n" + "=" * 50)
print("✅ ENVIRONMENT WORKING!")
print("=" * 50)