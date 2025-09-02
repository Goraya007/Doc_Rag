from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from config import settings
import torch

class LLMClient:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def initialize(self):
        """Initialize LLM with 4-bit quantization"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.LLM_MODEL,
            cache_dir=settings.CACHE_DIR
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=settings.CACHE_DIR
        )
        
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=settings.MAX_NEW_TOKENS,
            temperature=settings.TEMPERATURE,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True
        )
        
        return HuggingFacePipeline(pipeline=self.pipeline)

    def create_qa_chain(self, vector_store):
        """Create RAG QA chain with custom prompt"""
        prompt_template = """
        <|system|>
        You are an expert document analyst. Use the context below to answer the question.
        If unsure, say "I don't know". Provide detailed, professional responses.
        
        Context: {context}</s>
        <|user|>
        {question}</s>
        <|assistant|>
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.initialize(),
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": settings.TOP_K}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )