import langchain_groq as ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import streamlit as st
import time
from dotenv import load_dotenv

os.environ['GROQ_API_KEY'] =""
groq_api_key = os.environ['GROQ_API_KEY']
load_dotenv()

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader=WebBaseLoader("https://abeshith-portfolio.netlify.app/")
    st.session_state.docs=st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=100)
    st.session_state.splitted_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectorstore = FAISS.from_documents(st.session_state.splitted_docs,st.session_state.embeddings)

llm = ChatGroq(groq_api_key = groq_api_key,
               model_name="Gemma-7b-it")


prompt = ChatPromptTemplate.from_template(
    """
Answer the following question based only on the provided context
    Analyse the provided context and step by step properly and answer the question
    And based on the answer i'll tip you some amount of money.
     <context>
    {context}
    </context>
    Question :{input}
"""
)

chain = create_stuff_documents_chain(llm,prompt)
retriever = st.session_state.vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever,chain)

prompt = st.text_input("Input",key="input")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({'input':prompt})
    print("Response time:",time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Doc Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)