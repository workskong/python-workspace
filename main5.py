import os
import streamlit as st
import tiktoken
import time
import faiss

from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory


if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

store = LocalFileStore(".cache/embeddings")


def main():
    st.set_page_config(
    page_title="LLAMA3 & RAG",
    page_icon=":books:")

    st.title("LLAMA3 & RAG")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader(
            "File Uplaod",
            type=["pdf", "pptx", "docx"],
            accept_multiple_files=True
            )
    
        uploadFiles = st.button("Upload Files")
        
        # files = os.listdir(f"./.cache/files/")
        # selected_file = st.sidebar.selectbox(
        #     "Document Library",
        #     files,
        #     index=None,
        #     placeholder="All",
        # )
        # if selected_file:
        #     st.write("You selected:", selected_file)

    if uploadFiles:
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
     
        st.session_state.conversation = get_conversation_chain1(vetorestore)

        if 'messages' not in st.session_state:
            st.session_state['messages'] = [{"role": "assistant",  "content": "업로드 된 문서를 기반으로 질문에 답변을 합니다."}]

        # Example: Iterating through messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        history = StreamlitChatMessageHistory(key="chat_messages")
   
    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):

        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            start_time = time.time()

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    
                end_time = time.time()
                elapsed_time = end_time - start_time 
                st.write(f"Genarating took {elapsed_time:.2f} seconds")

# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def get_text(docs):

    doc_list = []
    
    for doc in docs:
        
        file_name = f"./.cache/files/{doc.name}"
        with open(file_name, "wb") as f:
            f.write(doc.getvalue())
        logger.info(f"./.cache/files/{file_name}")        

        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)  

    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    start_time = time.time()
    
    embeddings = HuggingFaceEmbeddings(
                                        model_name="BAAI/bge-m3",
                                        # model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cuda'},
                                        encode_kwargs={'normalize_embeddings': True}
                                    )

    cache_dir = LocalFileStore(f"./.cache/embeddings/cache")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectordb = FAISS.from_documents(text_chunks, cached_embeddings)
    
    end_time = time.time()
    elapsed_time = end_time - start_time 
    st.write(f"Embedding took {elapsed_time:.2f} seconds")

    return vectordb


def get_conversation_chain1(vetorestore):
    llm = ChatOllama(model="llama3", temperature=0.1)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type='mmr', verbose=True, top_k=2),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain


if __name__ == '__main__':
    main()