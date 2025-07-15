import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, 'chroma_meta')

if not os.path.exists(persistent_directory):
    print(f'the directory {persistent_directory} does not exist. Initializing Vector Store...')
    if not os.path.exists(books_dir):
        raise FileNotFoundError(f'the directory {books_dir} doest not exist. Please check the path')
    
    book_files = [f for f in os.listdir(books_dir) if f.endswith('.txt')]

    documents = []
    for book_file in book_files:
        text_path = os.path.join(books_dir, book_file)
        loader = TextLoader(text_path, encoding='utf-8')
        docs = loader.load()
        for doc in docs:
            doc.metadata = {'Source': text_path}
            documents.append(doc)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persistent_directory )
    print('Created Vector store and persist vector')

else:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(embedding_function=embeddings, persist_directory=persistent_directory)
    query = "Who is Odysseus' wife?"
    retriever = db.as_retriever(search_type='similarity_score_threshold', search_kwargs={'k': 3, 'score_threshold': 0.4},)
    results = retriever.invoke(query)
    for i, result in enumerate(results, 1):
        print(f"Document {i}: \n {result.page_content} \n")
        print(f"{result.metadata.get('Source')} \n")

   


