import os
import streamlit as st
import tiktoken
import time

from loguru import logger

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import StreamlitChatMessageHistory

from langchain_community.callbacks import get_openai_callback

from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate


if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")
if not os.path.exists(".cache/files_temp"):
    os.mkdir(".cache/files_temp")
if not os.path.exists(".cache/faiss_index"):
    os.mkdir(".cache/faiss_index")

cache_faiss = ".cache/faiss_index"
files = os.listdir(f"./.cache/files/")
files_temp = os.listdir(f"./.cache/files_temp/")

st.set_page_config(
    page_title="workskong",
    page_icon=":books:")

with open(".venv/style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

st.title("무엇이든 물어보세요")

col1, col2 = st.columns(2)
with col1:
    with st.expander("서비스 소개"):
        st.caption(
            """
            - 인터넷을 사용해야 하는 유료 LLM API를 사용하지 않고
            상업적으로도 무료로 사용할 수 있는 로컬 LLM을 이용해서 만든 로컬 AI입니다.
            - 채팅을 위해 사용된 LLM은 LLama3이며,
            더 많은 기능을 지원하기 위해서 향 후 다양한 LLM 모델을 활용할 예정이예요.
            - 여기서 로컬 AI라는 말은 이 서비스가 제 개인 PC에서 실행된다는 의미입니다.
            너무 개인적인 문서나 저에게 알려지기 싫은 정보가 담긴 문서는 업로드하지 마세요.
            - 한글로 물어봐도, 답변을 줍니다. 하지만 가급적 영어로 물어보세요.
            - 개발과정에서 이미 올린 문서는 삭제될 수 있어요.
            - 문서 읽기 기능은 Retrieval Augmented Generation(RAG), 검색증강생성를 이용해서 답변합니다.
            - 500 Page 이상의 파일은 가급적 올리지 말아주세요. 참고로 1,400 Page 업로드 시간은 약 5분 내외입니다.
            - 아직 구현해야 할 게 많이 남아있어요.
            """
            )

with col2:
    with st.expander("누구를 위한 서비스인가요?"):
        st.caption(
            """
            - 내 대화가 ChatGPT나 Gemini 같은 인터넷 서비스로 저장되는 게 그냥 싫으신 분
            - 여러 문서를 파일로 읽어서 요약이 필요 하신 분
            - 파일로 된 문서내용을 기반으로 이메일을 쓰거나, 스케줄 작성이 필요한 분
            - 무제한으로 AI가 지어내는 아무 의미없는 대화가 하고 싶은 분
            - 할 일 없는 분
            """
            )

tab1, tab2 = st.sidebar.tabs(["💾 저장하기", "🔧 제어판",])

with tab2:
    st.caption("LLM 옵션을 조정 할 수 있어요")
    with st.expander("🌶️ temperature"):
        st.caption(
            """
            낮은 온도에서는 모델이 가장 가능성이 높은 단어를 
            선택하는 경향이 있어 보다 안전하고 예측 가능한 결과를 생성합니다. 
            반면 높은 온도에서는 모델이 덜 가능성이 높은 단어를 선택할 가능성이 높아져 
            더 창의적이고 독창적인 결과를 생성할 수 있습니다.
            """
            )
        temperature = st.slider('출력 텍스트에서 선택하는 단어의 다양성을 제어합니다.', 0, 10, 0)

    with st.expander("🕵️‍♀️ top_p"):
        st.caption(
            """
            값이 낮을수록 모델은 가장 가능성이 높은 단어만 고려하여 안전하고 
            예측 가능한 결과를 생성합니다. 반면 높은 값일수록 모델은 덜 가능성이 
            높은 단어를 선택할 수 있는 여유가 있어 
            더 창의적이고 독창적인 결과를 생성할 수 있습니다. 
            """
            )
        top_p = st.slider('출력 텍스트에서 선택하는 단어의 범위를 제어합니다.', 0, 10, 0)
    
    with st.expander("🤖 Length"):
        st.caption(
            """
            짧은 길이는 간결하고 요약된 결과를 생성하는 반면, 
            긴 길이는 더 상세하고 포괄적인 결과를 생성합니다.
            """
            )
        Length = st.slider('생성하는 텍스트의 길이를 제어합니다.', 0, 1000, 200)

llm = ChatOllama(
    model="llama3", 
    temperature=(temperature/10), 
    top_p=(top_p/10), 
    Length=Length)

def get_text(docs,mode):
    doc_list = []
   
    for doc in docs:
        if mode == "save":
            file_name = f"./.cache/files/{doc.name}"
        else:
            file_name = f"./.cache/files_temp/{doc.name}"

        with open(file_name, "wb") as f:
            f.write(doc.getvalue())
        logger.info(file_name)
        #logger.info(f"./.cache/files/{file_name}")

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

        if mode == "temp":
            os.remove(file_name)

    return doc_list

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=50,
        length_function=tiktoken_len
        )
    chunks = text_splitter.split_documents(text)
    return chunks

def embeddings():
    # model_name = model_name="jhgan/ko-sroberta-multitask"
    model_name = model_name="BAAI/bge-m3"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    hfe = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return hfe

def get_vectorstore(text_chunks,mode):
    start_time = time.time()

    if mode == "save":
        file_dir = f"./.cache/embeddings/cache"
    else:
        file_dir = f"./.cache/embeddings/cache_temp"

    cache_dir = LocalFileStore(file_dir)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings(), cache_dir)

    new = FAISS.from_documents(text_chunks, cached_embeddings)

    if mode == "save":
        try:
            old = FAISS.load_local(cache_faiss, 
                                embeddings(), 
                                allow_dangerous_deserialization=True)
            old.merge_from(new)
            old.save_local(cache_faiss)
            vs = old

        except RuntimeError as e:
            new.save_local(cache_faiss)
            vs = new
    else:
        vs = new

    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write(f"Embedding took {elapsed_time:.2f} seconds")

    return vs

def get_conversation_chain_rag(vetorestore):
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type='mmr', verbose=True, top_k=1),
        memory=ConversationBufferMemory(memory_key='chat_history', 
                                        return_messages=True, 
                                        output_key='answer'),
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

def get_conversation_chain():
    template = """
    You are AI assistant
    {chat_history}
    {question}
    """

    prompt = PromptTemplate.from_template(template)

    conversation_chain = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=ConversationBufferMemory(memory_key="chat_history"),
        input_key="question",
        output_key='answer')
    return conversation_chain

with tab1:
    st.caption("저장된 문서는 대화에 활용될 수 있어요")
    with st.expander("문서 올리기"):
        # File Upload
        uploaded_files = st.file_uploader(
            "먼저 파일을 선택해주세요",
            type=["pdf", "pptx", "docx"],
            accept_multiple_files=True
        )
        if upload_button := st.button("🏇 저장"):  
            if uploaded_files == []:
                st.toast("먼저 파일을 선택해주세요")
            else:
                with st.spinner('잠깐만요...'):
                    start_time = time.time()

                    files_text = get_text(uploaded_files,"save")
                    text_chunks = get_text_chunks(files_text)
                    vs = get_vectorstore(text_chunks,"save")
                
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    st.write(f"Loading took {elapsed_time:.2f} seconds")

                    st.session_state.conversation = get_conversation_chain_rag(vs)

        st.caption(
            """
            저장 버튼을 누르면, 서버에 파일과 임베딩결과가
            Vector Dabase에 저장되며 답변할 때 활용합니다.
            """
        )

tab1, tab2, tab3 = st.tabs(["👄 아무말이나 해보세요", 
                            "📘 문서에 있는 내용을 대신 읽어드려요", 
                            "📚 이미 읽은 문서를 뒤져볼 수 있어요"])

with tab1:
    st.write(
    """
    - 이미 학습된 내용을 바탕으로 아무말에 답해줄거예요.
    """
    )
with tab2:
    # Temporary File Upload
    st.write(
    """
    - 문서를 임시로 업로드하고 질문해보세요.
    > 여기서 업로드한 파일은 저장되지 않고 삭제됩니다. 
    > 방금 업로드한 문서를 근거로 답변 할 수 있습니다.
    """
    )
    st.caption(
    """
    - 서버에 저장하고 싶다면 왼쪽 사이드바에 있는 "문서 올리기" 후 "저장" 기능을 사용하세요.
    """
    )
    uploaded_files = st.file_uploader(
        "문서 올리기",
        type=["pdf", "pptx", "docx"],
        accept_multiple_files=True
    )
    if uploaded_files:
        with st.spinner('잠깐만요...'):
            start_time = time.time()

            files_text = get_text(uploaded_files,"temp")
            text_chunks = get_text_chunks(files_text)
            vs = get_vectorstore(text_chunks,"temp")
        
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write(f"Loading took {elapsed_time:.2f} seconds")
            st.session_state.conversation = get_conversation_chain_rag(vs)           

with tab3:
    st.write(
    """
    - 저장된 문서를 뒤져볼까요?
    > 불러오기 버튼을 누르면 저장된 Vector Table을 읽어서 답변할 때 활용합니다.
    > 저장된 문서를 기반해서 답변을 하지만 결과를 너무 믿진 마세요.
    > 불러온 파일 정보는 아래를 참고하세요.
    """
    )
    st.caption(
    """
    - 서버에 저장하고 싶다면 왼쪽 사이드바에 있는 "문서 올리기" 후 "저장" 기능을 사용하세요.
    """
    )
    # Upload Database
    # selected_doc = st.selectbox(
    #    "Document Library",
    #    files,
    #    index=None,
    #    placeholder="All",
    #    key="selected_doc"
    #    )
    #
    #if selected_doc:
    #    st.write("You selected:", selected_doc)
    #    if 'selected_doc' not in st.session_state:
    #            st.session_state['selected_doc'] = selected_doc
    with st.expander("📘 저장된 문서 열람"):
        for file in files:
            st.caption(file)

    if load_db := st.button("불러오기"):
        start_time = time.time()

        vs = FAISS.load_local(cache_faiss, embeddings(), allow_dangerous_deserialization=True)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"Loading took {elapsed_time:.2f} seconds")
        st.session_state.conversation = get_conversation_chain_rag(vs)

def init_chat_input():
    if 'refLoad' not in st.session_state:
        st.session_state['refLoad'] = False
        chat_input="💬 아직 참고문서를 불러오지 않으셨네요. 왼쪽 사이드바에서 참고문서를 불러오거나 임시로 파일을 올려보세요. 물론 참고문서가 없어도 대화가 가능해요."
    else:
        chat_input="💬 참고문서 불러오기가 성공했어요. 이제 대화에서 참고문서에 있는 정보를 활용할거예요."
    return chat_input

# Chat logic
if query := st.chat_input("Enter a prompt here"):

    if 'conversation' not in st.session_state:
        st.session_state.conversation = get_conversation_chain()

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                        "content": "질문에 답변을 드립니다."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        chain = st.session_state.conversation

        start_time = time.time()

        # topic = st.session_state['selected_doc']
        query = (f"{query}")

        with st.spinner("생각중..."):
            result = chain({"question": query})
            with get_openai_callback() as cb:
                st.session_state.chat_history = result['chat_history']
            response = result['answer']
            if 'source_documents' in result:
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참조한 문서"):
                    for i, source_doc in enumerate(source_documents):
                        st.markdown(f"Source: {source_doc.metadata['source']}", 
                                    help=source_doc.page_content)
            else:
                st.markdown(response)

            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

        end_time = time.time()
        elapsed_time = end_time - start_time 
        st.write(f"Genarating took {elapsed_time:.2f} seconds")
