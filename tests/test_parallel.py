import requests
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the API endpoint
url = "http://localhost:7000/api/v1/predict"  # Replace with your actual endpoint


# Construct the payload according to your schema
payload = {
    "inputs": [
        {
            "FarmerId": 123,               # Replace with actual ID or None
            "TgtLang": "hin_Deva",               # Replace with target language code or None
            "Text": "Hello, how are you?",  # Replace with the actual text to translate
        }
    ]
}



def send_request(i):
    response = requests.post(url, json=payload)
    return i, response

if __name__ == "__main__":
    num_requests = 10
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(send_request, i) for i in range(num_requests)]
        for future in as_completed(futures):
            i, response = future.result()
            print(f"Response {i+1}: {response.status_code}")
            if response.status_code == 200:
                print("Response:", response.json())
            else:
                print("Error:", response.text)
