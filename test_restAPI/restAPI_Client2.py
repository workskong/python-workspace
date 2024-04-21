import requests

url = 'http://localhost:5000/generate_text'  
data = {'input': 'Write a sentence.'}

response = requests.post(url, json=data)  # POST 요청 보내기

if response.status_code == 200:
    # Get the JSON response
    response_data = response.json()

    # Print the entire result from RemoteRunnable
    print("RemoteRunnable Result:", response_data)
else:
    print("Error:", response.status_code)
