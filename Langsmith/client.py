import requests
import streamlit as st

def get_ollama_response(input_text):
    response = requests.post("http://localhost:8000/joke/invoke",
                             json={'input':{'topic':input_text}})
   
    return response.json()['output']
input_text = st.text_input('Input',key='input')

if input_text:
    st.write(get_ollama_response(input_text))