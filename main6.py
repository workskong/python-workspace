from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


cache_file = ".cache/faiss_index"

tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

model_name = model_name="BAAI/bge-m3" # "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
ko = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
"""
loader = PyPDFLoader(".cache/files/LGBR_Report_240214_20245819135853848.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=1000,
                    length_function = tiktoken_len
                )
docs = text_splitter.split_documents(pages)

db = FAISS.from_documents(docs, ko)

db.save_local(cache_file)


query = "인공지능 산업구조는 어떻게 구성되어있어?"
docs = db.similarity_search(query)
print(docs[0].page_content)
docs_and_scores = db.similarity_search_with_score(query)
"""

new_db = FAISS.load_local(".cache/faiss_index", ko, allow_dangerous_deserialization=True)

"""
query = "인공지능 산업구조는 어떻게 구성되어있어?"
docs = new_db.similarity_search_with_relevance_scores(query, k=3)
print("질문: {} \n".format(query))
for i in range(len(docs)):
    print("{0}번째 유사 문서 유사도 \n{1}".format(i+1,round(docs[i][1],2)))
    print("-"*100)
    print(docs[i][0].page_content)
    print("\n")
    print(docs[i][0].metadata)
    print("-"*100)
"""
    
llm = ChatOllama(model="llama3", temperature=0.1)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=new_db.as_retriever(search_type='mmr', verbose=True, top_k=9),
    memory=ConversationBufferMemory(memory_key='chat_history', 
                                    return_messages=True, 
                                    output_key='answer'),
    get_chat_history=lambda h: h,
    return_source_documents=True,
    verbose=True
)

query = input("Your question: ")
result = conversation_chain({"question": query})
response = result['answer']

# Inside your Streamlit application
print(result)
print(response)
