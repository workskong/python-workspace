from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate  # Import from prompts module
from langserve import RemoteRunnable
from flask import Flask, request, jsonify

app = Flask(__name__)

llm = RemoteRunnable("https://accurate-inviting-fowl.ngrok-free.app/llm/")

# Simulated data store (replace with your actual data storage)
data_store = {
    "1": "This is some text data with ID 1.",
    "2": "This is another piece of text data with ID 2."
}

@app.route('/texts', methods=['GET'])
def get_texts():
    # Simulate filtering based on query parameters
    filters = request.args.get('filter')
    filtered_data = {}
    if filters:
        # Implement filtering logic based on your needs
        # (e.g., filter by ID, content substring match)
        for key, value in data_store.items():
            if filters in value:
                filtered_data[key] = value
    else:
        filtered_data = data_store.copy()

    # Return the data as JSON with OData v4 formatting (consider using a library)
    return jsonify({
        "@odata.context": "$metadata#texts",
        "value": list(filtered_data.values())
    })

@app.route('/texts/<text_id>', methods=['GET'])
def get_text_by_id(text_id):
    if text_id in data_store:
        return jsonify({
            "@odata.context": "$metadata#texts/$entity",
            "ID": text_id,
            "Content": data_store[text_id]
        })
    else:
        return jsonify({"error": "Text not found"}), 404

@app.route('/texts', methods=['POST'])
def create_text():
    # Extract input text from the request body
    input_text = request.json.get('input')

    # Check for missing input text
    if not input_text:
        return jsonify({"error": "Missing required field: input"}), 400  # Bad request

    # Prepare the prompt with the input text
    prompt = ChatPromptTemplate.from_template(
        f"you are sap s4h expert:\n{input_text}"
    )

    # Build the chain with llm and StrOutputParser
    chain = prompt | llm | StrOutputParser()

    # Generate text using the chain
    output_text = chain.invoke({"input": "is the worth it?"})

    # Create a new text entry with ID and generated content
    new_text_id = str(len(data_store) + 1)
    new_text_content = output_text

    # Add the new text entry to the data store (replace with your actual storage)
    data_store[new_text_id] = new_text_content

    # Return the newly created text data with OData v4 formatting
    return jsonify({
        "@odata.context": "$metadata#texts/$entity",
        "ID": new_text_id,
        "Content": new_text_content
    }), 201  # Created status code

# Implement additional methods for OData operations like UPDATE and DELETE
# based on your specific data management needs

if __name__ == '__main__':
    app.run(debug=True)
