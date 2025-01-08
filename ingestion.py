import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
embeddings = OllamaEmbeddings(model="llama3.1:8b")
vector_store = Chroma(embedding_function=embeddings, persist_directory=os.environ['CHROMA_PATH'])

# loader = TextLoader("./mbcet_website_data.txt")
loader = JSONLoader(
    file_path='./sample.json',
    jq_schema='.[]',
    content_key=".content",
    is_content_key_jq_parsable=True,
    metadata_func=metadata_func
)
print("loader=", loader)
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
