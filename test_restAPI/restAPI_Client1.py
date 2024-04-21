import requests

url = 'http://localhost:5000/generate_text'  # 엔드포인트 URL
data = {'input': 'Write a sentence.'}  # 요청할 데이터

response = requests.post(url, json=data)  # POST 요청 보내기

if response.status_code == 200:
    result = response.json()
    output_text = result['output']
    print("Generated text:", output_text)
else:
    print("Error:", response.status_code)
