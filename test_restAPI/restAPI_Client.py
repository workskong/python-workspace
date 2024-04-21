import requests

url = 'http://localhost:5000/texts'  # Flask application URL
data = {'input': 'explain acdoca table'}  # Input text

# Send a POST request with the correct content type header
response = requests.post(url, json=data, headers={'Content-Type': 'application/json'})

if response.status_code == 201:  # Created status code
    result = response.json()
    text_id = result['ID']
    text_content = result['Content']
    print("New text created:")
    print("ID:", text_id)
    print("Content:", text_content)
else:
    print("Error:", response.status_code)
