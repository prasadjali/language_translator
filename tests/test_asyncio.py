
import asyncio
import aiohttp
import json

url = "http://localhost:7000/api/v1/predict"

payload = {
    "inputs": [
        {
            "FarmerId": 123,
            "TgtLang": "hin_Deva",
            "Text": "Hello, how are you?",
        }
    ]
}

async def send_request(session, i):
    async with session.post(url, json=payload) as response:
        status = response.status
        try:
            data = await response.json()
        except Exception:
            data = await response.text()
        return i, status, data

async def main():
    num_requests = 10
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i) for i in range(num_requests)]
        for future in asyncio.as_completed(tasks):
            i, status, data = await future
            print(f"Response {i+1}: {status}")
            if status == 200:
                print("Response:", data)
            else:
                print("Error:", data)

if __name__ == "__main__":
    asyncio.run(main())