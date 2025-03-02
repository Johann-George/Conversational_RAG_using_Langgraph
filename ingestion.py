import os
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, JSONLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def metadata_func(record: dict, metadata: dict):
    metadata_dict = record.get("metadata")
    # print(metadata_dict)
    if not isinstance(metadata_dict, dict):
        metadata_dict = {}

    title = metadata_dict.get("title", "No Title found")
    sourceurl = metadata_dict.get("sourceURL", "No source url found")

    metadata["title"] = str(title)
    metadata["source"] = str(sourceurl)
    return metadata


llm = ChatOllama(model="llama3.1:8b")
embeddings = OllamaEmbeddings(model="chroma/all-minilm-l6-v2-f32")
vector_store = Chroma(embedding_function=embeddings, persist_directory=os.environ['CHROMA_PATH'])
# index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
#
# vector_store = FAISS(
#     embedding_function=embeddings,
#     index = index,
#     docstore=InMemoryDocstore(),
#     index_to_docstore_id={},
# )

# loader = PyPDFLoader("./refined.pdf")
loader = JSONLoader(
    file_path='./sample.json',
    jq_schema='.[]',
    content_key="markdown",
    metadata_func=metadata_func
)

docs = loader.load()
# print("docs=", docs)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False
)

all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)
# vector_store.add_documents(documents=all_splits)
# vector_store.save_local("faiss_index")
# PineconeVectorStore.from_documents(all_splits, embeddings, index_name=os.environ['INDEX_NAME'])
