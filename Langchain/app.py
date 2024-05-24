import langchain
from dotenv import load_dotenv
import os
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()

os.environ['LANGCHAIN_TRACING_V2']= 'true'
os.environ['LANGCHAIN_API_KEY'] = ''

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer."),
    ("user", "{input}")
])

llm = Ollama(model="llama2")

input_text = st.text_input('Input',key='input')


output_parser = StrOutputParser()
chain = prompt | llm | output_parser
#chain.invoke({"input": "how can langsmith help with testing?"})

if input_text:
    st.write(chain.invoke({"input":input_text}))

