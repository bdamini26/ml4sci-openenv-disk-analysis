import requests

url = "https://damsaychesse-disk-image-analysis.hf.space/step"
action = {
    "analysis_type": "detect_transit",
    "confidence": 0.85,
    "prediction": "yes_transit",
    "reasoning": "test"
}

response = requests.post(url, json=action)
print(response.json())