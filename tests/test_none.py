import requests
import json
import os

# Define the API endpoint
url = "http://localhost:7000/api/v1/predict"  # Replace with your actual endpoint


# Construct the payload according to your schema
payload = {
    "inputs": [
        {
            "FarmerId": 123,               # Replace with actual ID or None
            "TgtLang": "hin_Deva",               # Replace with target language code or None
            "Text": None
        }
    ]
}

# Send the POST request
response = requests.post(url, json=payload)

# Handle the response
if response.status_code == 200:
    print("✅ Success!")
    print("Response:", response.json())
else:
    print("❌ Failed with status code:", response.status_code)
    print("Response:", response.text)
