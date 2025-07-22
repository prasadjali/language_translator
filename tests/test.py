import requests
import json
import os

# Define the API endpoint
url = "http://localhost:7000/api/v1/predict"  # Replace with your actual endpoint

input_sentences = ["""{
    "Disease Name": "Powdery Mildew",
    "Local Name": "Dhan Ki Kharash",
    "Disease Description": "Powdery mildew is a fungal disease that affects a variety of plants, including rice. It is characterized by the presence of white, powdery fungal growth on the leaves, stems, and sometimes flowers. The disease thrives in warm, dry conditions and can lead to reduced photosynthesis, stunted growth, and lower yields if not managed properly.",
    "Symptoms": [
        "White powdery spots on leaves",
        "Yellowing of leaves",
        "Leaf curling or distortion",
        "Premature leaf drop"
    ],
    "Causes": [
        "Fungal pathogens, primarily from the genus Erysiphe",
        "High humidity and temperature fluctuations",
        "Poor air circulation around plants",
        "Overcrowding of plants"
    ],
    "Treatment": {
        "Chemical Control": [
            "Fungicides containing active ingredients such as sulfur, myclobutanil, or triadimefon",
            "Systemic fungicides for severe infections"
        ],
        "Biological Control": [
            "Application of beneficial fungi such as Trichoderma spp.",
            "Use of biopesticides containing Bacillus subtilis"
        ],
        "Cultural Control": [
            "Improving air circulation by spacing plants adequately",
            "Removing and destroying infected plant debris",
            "Practicing crop rotation to reduce pathogen load",
            "Avoiding overhead irrigation to reduce humidity"
        ]
    },
    "Prevention": [
        "Selecting resistant rice varieties",
        "Implementing proper irrigation practices to avoid excess moisture",
        "Regular monitoring for early signs of infection",
        "Maintaining healthy soil and plant nutrition"
    ]
}"""]


parsed_text = json.loads(input_sentences[0])


# Construct the payload according to your schema
payload = {
    "inputs": [
        {
            "FarmerId": 123,               # Replace with actual ID or None
            "TgtLang": "hin_Deva",               # Replace with target language code or None
            "Text": json.dumps(parsed_text, ensure_ascii=False, indent=2)
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
