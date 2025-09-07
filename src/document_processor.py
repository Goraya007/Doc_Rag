from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import settings
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    @staticmethod
    def load_document(file_path: str):
        """Load document based on file extension"""
        from pathlib import Path
        ext = Path(file_path).suffix.lower()
        
        loader_map = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader
        }
        
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")
        
        try:
            loader = loader_map[ext](file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

    @staticmethod
    def chunk_documents(documents):
        """Split documents into optimized chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            add_start_index=True,
            length_function=len
        )
        return text_splitter.split_documents(documents)