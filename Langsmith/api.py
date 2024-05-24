from fastapi import FastAPI
from langserve import add_routes
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['LANGCHAIN_TRACING_V2']= 'true'
os.environ['LANGCHAIN_API_KEY'] = ''

app = FastAPI(
    title="Langchain server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

llm = Ollama(model="llama2")

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(
    app,
    prompt|llm,
    path='/joke',
)

if __name__ == "__main__":
    uvicorn.run(app,host="localhost",port=8000)