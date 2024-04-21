from flask import Flask, request, jsonify
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langserve import RemoteRunnable

app = Flask(__name__)

llm = RemoteRunnable("https://accurate-inviting-fowl.ngrok-free.app/llm/")

@app.route('/generate_text', methods=['POST'])
def generate_text():
    input_text = request.json.get('input')

    prompt = ChatPromptTemplate.from_template(
        f"I am a woman:\n{input_text}"
    )

    chain = prompt | llm | StrOutputParser()

    output_text = chain.invoke({"input": "is the worth it?"})

    return jsonify({'output': output_text})

if __name__ == '__main__':
    app.run(debug=True)
