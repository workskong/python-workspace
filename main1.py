import os
import streamlit as st
import time  # 시간 모듈 추가
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import ChatMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.vectorstores.faiss import FAISS
from langserve import RemoteRunnable

USE_BGE_EMBEDDING = True

if not USE_BGE_EMBEDDING:
    os.environ["OPENAI_API_KEY"] = ""

LANGSERVE_ENDPOINT = "https://accurate-inviting-fowl.ngrok-free.app/llm/"

if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

RAG_PROMPT_TEMPLATE = """You are an AI that answers questions. Search and then answer the questions using context. If you can't find the answer, do not answer!.
Question: {question} 
Context: {context} 
Answer:"""

st.set_page_config(page_title="llama3 Local", page_icon="💬")
st.title("llama3 Local")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="system", content="What can i do for you?")
    ]

def print_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    else:
        for msg in st.session_state.messages:
            st.chat_message(msg.role).write(msg.content)

def add_history(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    start_time = time.time()  # 시작 시간 측정

    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        length_function=len,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=text_splitter)

    if USE_BGE_EMBEDDING:
        model_name = "jhgan/ko-sroberta-multitask"
        model_kwargs = {
            # "device": "cuda"
            # "device": "mps"
            "device": "cpu"
        }
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    else:
        embeddings = []
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, embedding=cached_embeddings)
    retriever = vectorstore.as_retriever()

    end_time = time.time()  # 종료 시간 측정
    elapsed_time = end_time - start_time  # 소요된 시간 계산
    st.write(f"Embedding took {elapsed_time:.2f} seconds")  # 시간 출력

    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

with st.sidebar:
    file = st.file_uploader(
        "File Uplaod",
        type=["pdf", "pptx", "xlsx", "docx", "csv", "txt"],
    )

if file:
    retriever = embed_file(file)

print_history()

if user_input := st.chat_input():
    add_history("user", user_input)
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        ollama = RemoteRunnable(LANGSERVE_ENDPOINT)
        chat_container = st.empty()
        if file is not None:
            prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
                | prompt
                | ollama
                | StrOutputParser()
            )

            answer = rag_chain.stream(user_input) 
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))
            add_history("ai", "".join(chunks))
        else:
            prompt = ChatPromptTemplate.from_template(
                "Please answer the following questions at length:\n{input}"
            )

            chain = prompt | ollama | StrOutputParser()

            answer = chain.stream(user_input)
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))
            add_history("ai", "".join(chunks))
