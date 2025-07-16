import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import (TextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter, SentenceTransformersTokenTextSplitter)


load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'books', 'odyssey.txt')
db_dir = os.path.join(current_dir, 'db')

if not os.path.exists(file_path):
    raise FileNotFoundError(f'the file does not exist. Please check the path')

print("Text Loading....")
loader = TextLoader(file_path, encoding='utf-8')
documents = loader.load()
print('---Finish loading---')

# embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

def create_vector_store(docs, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print('The directory does not exist. Initializing Vector Store...')
        db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persistent_directory)
    else:
        print('the directory already exists. No need to initialize')

# CharacterTextSplitter
char_loader = CharacterTextSplitter(separator="\n\n", chunk_size=1000, chunk_overlap=20)
doc_char = char_loader.split_documents(documents)

recur_loader = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=1000, chunk_overlap=20)
doc_recur = recur_loader.split_documents(documents)

def query_vector(query,store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print('The directory does not exist. Please check the path')
    else:
        # embeddings = OpenAIEmbeddings(model='text-embedding-3-small')  
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        db = Chroma(embedding_function=embeddings, persist_directory=persistent_directory)
        retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 2, "score_threshold": 0.1})
        relevant_docs = retriever.invoke(query)
        for i, doc in enumerate(relevant_docs, 1):
            print(f"\n Document {i}: \n {doc.page_content}\n")
            if doc.metadata:
                print(f"{doc.metadata.get('Source', 'Unknow')}")

create_vector_store(doc_char, "chromadb_recursvie")
query = "Who is Odysseus' wife?"

query_vector(query,"chromadb_recursvie" )