import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatOllama(model="llama3.2:3b")
embeddings = OllamaEmbeddings(model="llama3.2:3b")
vector_store = Chroma(embedding_function=embeddings, persist_directory=os.environ['CHROMA_PATH'])

loader = TextLoader("./mbcet_website_data.txt")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

