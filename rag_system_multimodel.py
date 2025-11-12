"""
Enhanced RAG System with Multi-Model Support
Supports: Claude (Anthropic), Gemini (Google), GPT (OpenAI)
"""

import os
from typing import Dict, List
from pathlib import Path


class RAGSystemMultiModel:
    """RAG System supporting multiple LLM providers"""
    
    def __init__(self, pdf_path: str, embedding_model: str, 
                 model_provider: str, model_name: str, api_key: str):
        """
        Initialize RAG system with multi-model support
        
        Args:
            pdf_path: Path to PDF document
            embedding_model: Embedding model name
            model_provider: 'Claude', 'Google Gemini', or 'OpenAI'
            model_name: Specific model to use
            api_key: API key for the provider
        """
        self.pdf_path = pdf_path
        self.embedding_model_name = embedding_model
        self.model_provider = model_provider
            self.model_name = model_name
        self.api_key = api_key
        
        self.document_processor = None
        self.embedding_manager = None
        self.retriever = None
        self.generator = None
        self.chunks = None
        self.embeddings = None
        
        print(f"✓ RAG System initialized with {model_provider} - {model_name}")
    
    def build(self, chunk_size: int = 500, chunk_overlap: int = 50, use_cached: bool = False):
        """Build the RAG system"""
        print(f"\n{'='*70}")
        print("BUILDING RAG SYSTEM")
        print(f"{'='*70}\n")
        
        # Step 1: Load and chunk document
        print("[STEP 1/4] Processing Document")
        print("-" * 70)
        
        try:
            from src.rag_system import DocumentProcessor
            self.document_processor = DocumentProcessor(self.pdf_path)
            self.document_processor.load_pdf()
            self.chunks = self.document_processor.chunk_document(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            self.document_processor.save_chunks('data/chunks.json')
        except Exception as e:
            print(f"Error in document processing: {e}")
            raise
        
        # Step 2: Create embeddings
        print("\n[STEP 2/4] Creating Embeddings")
        print("-" * 70)
        
        try:
            from src.rag_system import EmbeddingManager
            self.embedding_manager = EmbeddingManager(self.embedding_model_name)
            
            if use_cached and Path('data/embeddings.pkl').exists():
                self.embeddings = EmbeddingManager.load_embeddings('data/embeddings.pkl')
            else:
                self.embeddings = self.embedding_manager.create_embeddings(self.chunks)
                self.embedding_manager.save_embeddings('data/embeddings.pkl')
        except Exception as e:
            print(f"Error in embedding: {e}")
            raise
        
        # Step 3: Initialize retriever
        print("\n[STEP 3/4] Setting up Retriever")
        print("-" * 70)
        
        try:
            from src.rag_system import Retriever
            self.retriever = Retriever(
                self.chunks,
                self.embeddings,
                self.embedding_manager.model
            )
            print("✓ Retriever ready")
        except Exception as e:
            print(f"Error in retriever setup: {e}")
            raise
        
        # Step 4: Initialize generator based on provider
        print("\n[STEP 4/4] Setting up Generator")
        print("-" * 70)
        
        self._initialize_generator()
        
        print("\n" + "="*70)
        print("✓ RAG SYSTEM READY!")
        print("="*70 + "\n")
    
    def _initialize_generator(self):
        """Initialize generator based on model provider"""
        
        if self.model_provider == 'Claude':
            self._setup_claude()
        
        elif self.model_provider == 'Google Gemini':
            self._setup_gemini()
        
        elif self.model_provider == 'OpenAI':
            self._setup_openai()
    
    def _setup_claude(self):
        """Setup Anthropic Claude"""
        try:
            from anthropic import Anthropic
            
            self.client = Anthropic(api_key=self.api_key)
            self.provider_type = 'claude'
            print(f"✓ Claude Generator ready ({self.model_name})")
        
        except Exception as e:
            print(f"Error setting up Claude: {e}")
            raise
    
    def _setup_gemini(self):
        """Setup Google Gemini"""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
            self.provider_type = 'gemini'
            print(f"✓ Gemini Generator ready ({self.model_name})")
        
        except Exception as e:
            print(f"Error setting up Gemini: {e}")
            raise
    
    def _setup_openai(self):
        """Setup OpenAI GPT"""
        try:
            from openai import OpenAI
            
            self.client = OpenAI(api_key=self.api_key)
            self.provider_type = 'openai'
            print(f"✓ OpenAI Generator ready ({self.model_name})")
        
        except Exception as e:
            print(f"Error setting up OpenAI: {e}")
            raise
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        End-to-end RAG query with selected model
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
        
        Returns:
            Dictionary with answer and metadata
        """
        print(f"\n📝 QUERY: {question}\n")
        print("[RETRIEVE] Searching for relevant content...")
        
        # Retrieve
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        
        # Prepare context
        context = "\n\n".join([
            f"[Page {r.page}, Similarity: {r.similarity:.2f}]\n{r.text}"
            for r in retrieved
        ])
        
        # Generate with selected provider
        print("[GENERATE] Creating answer...")
        
        if self.provider_type == 'claude':
            answer = self._generate_claude(question, context)
        
        elif self.provider_type == 'gemini':
            answer = self._generate_gemini(question, context)
        
        elif self.provider_type == 'openai':
            answer = self._generate_openai(question, context)
        
        result = {
            'question': question,
            'retrieved_results': retrieved,
            'context': context,
            'answer': answer,
            'metadata': {
                'retrieval_count': len(retrieved),
                'embedding_model': self.embedding_model_name,
                'model_provider': self.model_provider,
                'model_name': self.model_name
            }
        }
        
        return result
    
    def _generate_claude(self, question: str, context: str) -> str:
        """Generate answer using Claude"""
        try:
            system_prompt = """You are an expert on Apple's organizational structure 
and innovation practices. Based on the provided context from an Apple case study, 
answer the user's question accurately and thoroughly.

Guidelines:
- Ground your answer in the provided context
- Be specific and detailed
- Avoid speculation beyond what's in the context
- Cite relevant details from the case study
- If context doesn't contain answer, say so clearly"""
            
            user_message = f"""CONTEXT FROM APPLE CASE STUDY:
{context}

QUESTION:
{question}

Please provide a detailed answer based on the context above."""
            
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            return response.content[0].text
        
        except Exception as e:
            print(f"✗ Error generating answer with Claude: {e}")
            raise
    
    def _generate_gemini(self, question: str, context: str) -> str:
        """Generate answer using Google Gemini"""
        try:
            prompt = f"""You are an expert on Apple's organizational structure and innovation practices.

CONTEXT FROM APPLE CASE STUDY:
{context}

QUESTION:
{question}

Please provide a detailed answer based on the context above. Be specific, cite details, and avoid speculation."""
            
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 1000,
                    "temperature": 0.7,
                }
            )
            
            return response.text
        
        except Exception as e:
            print(f"✗ Error generating answer with Gemini: {e}")
            raise
    
    def _generate_openai(self, question: str, context: str) -> str:
        """Generate answer using OpenAI GPT"""
        try:
            system_prompt = """You are an expert on Apple's organizational structure 
and innovation practices. Based on the provided context from an Apple case study, 
answer the user's question accurately and thoroughly.

Guidelines:
- Ground your answer in the provided context
- Be specific and detailed
- Avoid speculation beyond what's in the context
- Cite relevant details from the case study"""
            
            user_message = f"""CONTEXT FROM APPLE CASE STUDY:
{context}

QUESTION:
{question}

Please provide a detailed answer based on the context above."""
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"✗ Error generating answer with OpenAI: {e}")
            raise