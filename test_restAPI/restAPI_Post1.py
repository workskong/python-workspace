from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/generate_text', methods=['POST'])
def generate_text():
    output_text_set = {'output', 'text'}
    output_text_list = list(output_text_set)  # set을 list로 변환
    return jsonify({'output': output_text_list})  # list를 JSON으로 직렬화하여 반환

if __name__ == '__main__':
    app.run(debug=True)
