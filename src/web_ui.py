import gradio as gr
from src.document_processor import DocumentProcessor
from src.vector_store_manager import VectorStoreManager
from src.llm_client import LLMClient
import os

class WebUI:
    def __init__(self):
        self.qa_system = None
        
    def process_document(self, file):
        """Process uploaded document"""
        try:
            processor = DocumentProcessor()
            docs = processor.load_document(file.name)
            chunks = processor.chunk_documents(docs)
            
            vector_mgr = VectorStoreManager()
            vector_store = vector_mgr.create_store(chunks)
            
            llm_client = LLMClient()
            self.qa_system = llm_client.create_qa_chain(vector_store)
            
            return " Document processed successfully!"
        except Exception as e:
            return f" Error: {str(e)}"

    def ask_question(self, question):
        """Handle question answering"""
        if not self.qa_system:
            return "Please process a document first", []
        
        result = self.qa_system({"query": question})
        
        sources = []
        for doc in result['source_documents']:
            sources.append({
                "content": doc.page_content[:200] + "...",
                "page": doc.metadata.get('page', 'N/A'),
                "source": os.path.basename(doc.metadata.get('source', ''))
            })
            
        return result['result'], sources

    def launch(self):
        """Launch Gradio interface"""
        with gr.Blocks(theme=gr.themes.Soft(), title="Document QA Pro") as ui:
            gr.Markdown("#  Advanced Document QA System")
            
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Upload Document")
                    process_btn = gr.Button("Process Document")
                    status = gr.Textbox(label="Status")
                
                with gr.Column():
                    question = gr.Textbox(label="Your Question")
                    ask_btn = gr.Button("Ask Question")
                    answer = gr.Textbox(label="Answer", lines=5)
                    sources = gr.JSON(label="Source References")
            
            process_btn.click(
                self.process_document, 
                inputs=file_input, 
                outputs=status
            )
            ask_btn.click(
                self.ask_question,
                inputs=question,
                outputs=[answer, sources]
            )
        
        ui.launch(server_port=7860, share=True)