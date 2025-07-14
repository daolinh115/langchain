import os
import shutil # Để xóa thư mục FAISS index cũ
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings # Dùng OpenAIEmbeddings để nhất quán
# from langchain_community.embeddings import HuggingFaceEmbeddings # Có thể dùng cái này để test thay thế OpenAIEmbeddings
from langchain_community.vectorstores import FAISS # Import thư viện FAISS
from dotenv import load_dotenv
import time
import logging
import sys

# Cấu hình logging để in ra console ngay lập tức
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout)
                    ])
sys.stdout.flush() 
logging.getLogger().handlers[0].flush = sys.stdout.flush

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
# Đường dẫn để lưu index FAISS (không phải là persistent_directory như Chroma)
faiss_index_path = os.path.join(current_dir, "db", 'faiss_index')

# Đảm bảo thư mục cha của faiss_index_path tồn tại
os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)

# --- PHẦN TẠO VÀ LƯU INDEX FAISS ---
# Kiểm tra nếu index FAISS đã tồn tại, xóa nó để chạy lại từ đầu cho mục đích test
if os.path.exists(faiss_index_path):
    logging.info(f"Existing FAISS index found at {faiss_index_path}. Deleting it for a clean run...")
    try:
        shutil.rmtree(faiss_index_path) # Xóa toàn bộ thư mục index FAISS
        logging.info("Old FAISS index deleted.")
    except Exception as e:
        logging.error(f"Could not delete old FAISS index: {e}. Please delete manually and retry.", exc_info=True)
        sys.exit(1)

logging.info("FAISS index does not exist or needs re-initialization. Initializing FAISS vector store...")
sys.stdout.flush()

if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path")

try:
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    logging.info(f"Loaded {len(documents)} documents from {file_path}")
    sys.stdout.flush()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    logging.info(f"Number of document chunks: {len(docs)}")
    logging.info(f"Sample content chunk:\n {docs[0].page_content}\n")
    sys.stdout.flush()

    logging.info('---Creating Embedding---')
    sys.stdout.flush()
    # Sử dụng OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    # Hoặc để test với HuggingFaceEmbeddings nếu nghi ngờ OpenAI
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    logging.info("---Finished creating embedding---")
    sys.stdout.flush()

    logging.info("---Creating FAISS Vector Store---")
    sys.stdout.flush()

    # Tạo FAISS index từ các tài liệu và embeddings
    db_faiss = FAISS.from_documents(docs, embeddings) 
    
    logging.info("FAISS.from_documents() returned successfully.")
    sys.stdout.flush()

    # Lưu FAISS index ra đĩa
    logging.info(f"Saving FAISS index to {faiss_index_path}...")
    db_faiss.save_local(faiss_index_path)
    logging.info(f"FAISS index saved to {faiss_index_path}")
    sys.stdout.flush()

    # Kiểm tra số lượng tài liệu trong FAISS (nội bộ, không phải từ disk)
    # Lưu ý: FAISS.index.ntotal cho biết số lượng vector trong index
    logging.info(f"Number of documents in FAISS index immediately after creation: {db_faiss.index.ntotal}")
    sys.stdout.flush()

    if db_faiss.index.ntotal > 0:
        logging.info("FAISS vector store created with documents. Running test query...")
        try:
            # Thử một truy vấn nhỏ
            test_query_result = db_faiss.similarity_search("What is the main character?", k=1)
            logging.info(f"Test query successful. Result count: {len(test_query_result)}")
            logging.info(f"Sample test query result: {test_query_result[0].page_content[:100]}...")
            sys.stdout.flush()
        except Exception as query_e:
            logging.error(f"Error during post-creation test query: {query_e}", exc_info=True)
            sys.stdout.flush()
    else:
        logging.warning("FAISS.from_documents completed but reported 0 documents after creation. This indicates a problem with data processing.")
        sys.stdout.flush()


    time.sleep(1) 
    logging.info("\n--- Finished creating and saving FAISS vector store ---") 
    sys.stdout.flush()

except Exception as e:
    logging.error(f'An unexpected error occurred during FAISS vector store creation: {e}', exc_info=True)
    sys.stdout.flush()
    sys.exit(1)

# --- PHẦN TẢI LẠI INDEX FAISS ĐỂ XÁC MINH ---
try:
    logging.info("\n--- Loading existing FAISS vector store (for verification/use) ---")
    sys.stdout.flush()
    # Tải lại index FAISS từ đường dẫn đã lưu
    # Cần truyền lại embedding function khi tải FAISS index
    db_loaded_faiss = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    
    # Kiểm tra số lượng tài liệu trong FAISS index đã tải
    loaded_doc_count = db_loaded_faiss.index.ntotal
    logging.info(f"Number of documents in the loaded FAISS vector store: {loaded_doc_count}")
    
    if loaded_doc_count > 0:
        logging.info("FAISS vector store loaded and contains data.")
        # Thử một truy vấn sau khi tải để xác nhận hoạt động
        test_query_result_loaded = db_loaded_faiss.similarity_search("Who is Odysseus?", k=1)
        logging.info(f"Post-load test query successful. Result count: {len(test_query_result_loaded)}")
        logging.info("FAISS vector store is ready for use.")
    else:
        logging.warning("FAISS vector store loaded but contains 0 documents. Data might not have been saved correctly during creation.")
    sys.stdout.flush()
except Exception as e:
    logging.error(f"Error loading or verifying FAISS vector store: {e}", exc_info=True)
    sys.stdout.flush()
    sys.exit(1)