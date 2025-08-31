from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import settings
import os

class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': settings.DEVICE},
            cache_folder=settings.CACHE_DIR
        )
        self.vector_store = None

    def create_store(self, documents):
        """Create FAISS vector store"""
        self.vector_store = FAISS.from_documents(
            documents, 
            self.embeddings
        )
        return self.vector_store

    def save_store(self, store_name="default"):
        """Persist vector store"""
        path = os.path.join(settings.VECTOR_STORE_DIR, store_name)
        self.vector_store.save_local(path)
        return path

    def load_store(self, store_name="default"):
        """Load existing vector store"""
        path = os.path.join(settings.VECTOR_STORE_DIR, store_name)
        self.vector_store = FAISS.load_local(
            path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        return self.vector_store