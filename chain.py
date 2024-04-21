from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="llama3")

prompt = ChatPromptTemplate.from_template("{topic} you are expert of SAP S4H")

chain = prompt | llm | StrOutputParser()
