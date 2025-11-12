"""
Complete RAG System Implementation
Apple Organizational Model Case Study

This is a COMPLETE, WORKING implementation.
Ready to customize and deploy.
"""

import json
import pickle
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path

# Dependencies to install:
# pip install pypdf langchain sentence-transformers chromadb anthropic

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from anthropic import Anthropic


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Chunk:
    """Represents a document chunk"""
    id: int
    text: str
    page: int
    chunk_num: int


@dataclass
class RetrievedResult:
    """Retrieved chunk with similarity score"""
    chunk_id: int
    text: str
    page: int
    similarity: float
    rank: int


# ============================================================================
# DOCUMENT LOADING & CHUNKING
# ============================================================================

class DocumentProcessor:
    """Handle PDF loading and chunking"""
    
    def __init__(self, pdf_path: str):
        """
        Initialize with PDF path
        
        Args:
            pdf_path: Path to Apple organization PDF
        """
        self.pdf_path = pdf_path
        self.raw_text = None
        self.chunks = []
    
    def load_pdf(self) -> str:
        """
        Load PDF and extract text
        
        Returns:
            Complete text from PDF
        """
        print(f"[LOADING] Reading PDF: {self.pdf_path}")
        
        try:
            reader = PdfReader(self.pdf_path)
            text = ""
            
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                # Add page marker for reference
                text += f"\n--- PAGE {page_num} ---\n{page_text}\n"
            
            self.raw_text = text
            print(f"✓ Successfully loaded PDF ({len(reader.pages)} pages)")
            return text
            
        except FileNotFoundError:
            print(f"✗ Error: File not found at {self.pdf_path}")
            raise
        except Exception as e:
            print(f"✗ Error reading PDF: {e}")
            raise
    
    def chunk_document(self, 
                      chunk_size: int = 500, 
                      chunk_overlap: int = 50) -> List[Chunk]:
        """
        Split document into chunks
        
        Args:
            chunk_size: Size of each chunk in tokens (approx)
            chunk_overlap: Overlap between chunks
        
        Returns:
            List of Chunk objects
        """
        if not self.raw_text:
            raise ValueError("Must load PDF first")
        
        print(f"[CHUNKING] Splitting into chunks (size={chunk_size}, overlap={chunk_overlap})")
        
        # Use RecursiveCharacterTextSplitter for better quality chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        text_chunks = splitter.split_text(self.raw_text)
        
        # Create Chunk objects with metadata
        chunks = []
        chunk_id = 0
        
        for chunk_num, text in enumerate(text_chunks):
            # Extract page number from text (if marked)
            page = self._extract_page_num(text)
            
            chunk = Chunk(
                id=chunk_id,
                text=text,
                page=page,
                chunk_num=chunk_num
            )
            chunks.append(chunk)
            chunk_id += 1
        
        self.chunks = chunks
        print(f"✓ Created {len(chunks)} chunks")
        return chunks
    
    def _extract_page_num(self, text: str) -> int:
        """Extract page number from chunk text"""
        if "--- PAGE" in text:
            try:
                start = text.index("--- PAGE ") + 9
                end = text.index("---", start)
                return int(text[start:end].strip())
            except:
                return 0
        return 0
    
    def save_chunks(self, filepath: str):
        """Save chunks to JSON for later use"""
        data = {
            'chunks': [asdict(c) for c in self.chunks],
            'metadata': {
                'total_chunks': len(self.chunks),
                'pdf_path': self.pdf_path
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved chunks to {filepath}")
    
    @staticmethod
    def load_chunks(filepath: str) -> List[Chunk]:
        """Load previously saved chunks"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return [Chunk(**c) for c in data['chunks']]


# ============================================================================
# EMBEDDING & VECTOR STORE
# ============================================================================

class EmbeddingManager:
    """Handle embeddings and vector storage"""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize embedding model
        
        Args:
            model_name: HuggingFace model for embeddings
                       Options: "all-mpnet-base-v2" (best quality)
                               "all-MiniLM-L6-v2" (faster)
        """
        print(f"[EMBEDDING] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embeddings = []
        print(f"✓ Model loaded")
    
    def create_embeddings(self, chunks: List[Chunk]) -> np.ndarray:
        """
        Create embeddings for all chunks
        
        Args:
            chunks: List of Chunk objects
        
        Returns:
            numpy array of embeddings (shape: num_chunks × embedding_dim)
        """
        print(f"[EMBEDDING] Creating embeddings for {len(chunks)} chunks...")
        
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        self.embeddings = embeddings
        print(f"✓ Created embeddings (shape: {embeddings.shape})")
        return embeddings
    
    def save_embeddings(self, filepath: str):
        """Save embeddings to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.embeddings, f)
        print(f"✓ Saved embeddings to {filepath}")
    
    @staticmethod
    def load_embeddings(filepath: str) -> np.ndarray:
        """Load previously saved embeddings"""
        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"✓ Loaded embeddings (shape: {embeddings.shape})")
        return embeddings


# ============================================================================
# RETRIEVAL
# ============================================================================

class Retriever:
    """Semantic search and retrieval"""
    
    def __init__(self, 
                 chunks: List[Chunk],
                 embeddings: np.ndarray,
                 embedding_model: SentenceTransformer):
        """
        Initialize retriever
        
        Args:
            chunks: List of document chunks
            embeddings: Pre-computed embeddings for chunks
            embedding_model: Model for encoding queries
        """
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedding_model = embedding_model
    
    def retrieve(self, 
                query: str, 
                top_k: int = 3,
                min_similarity: float = 0.3) -> List[RetrievedResult]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User query string
            top_k: Number of chunks to return
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of RetrievedResult objects ranked by relevance
        """
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            if similarities[idx] >= min_similarity:
                results.append(RetrievedResult(
                    chunk_id=idx,
                    text=self.chunks[idx].text,
                    page=self.chunks[idx].page,
                    similarity=float(similarities[idx]),
                    rank=rank
                ))
        
        return results


# ============================================================================
# GENERATION (LLM)
# ============================================================================

class Generator:
    """Generate answers using Claude"""
    
    def __init__(self, model: str = "claude-3-haiku-20240307"):
        """
        Initialize Claude client
        
        Args:
            model: Claude model to use
        """
        self.client = Anthropic()
        self.model = model
    
    def generate(self,
                query: str,
                context: str,
                system_prompt: str = None) -> str:
        """
        Generate answer using Claude
        
        Args:
            query: User question
            context: Retrieved context from documents
            system_prompt: Custom system prompt (optional)
        
        Returns:
            Generated answer
        """
        if not system_prompt:
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
{query}

Please provide a detailed answer based on the context above."""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            print(f"✗ Error generating answer: {e}")
            raise
    
    def generate_with_streaming(self, query: str, context: str) -> str:
        """Generate answer with streaming (optional)"""
        # For real-time output, implement streaming
        # For now, use regular generation
        return self.generate(query, context)


# ============================================================================
# COMPLETE RAG SYSTEM
# ============================================================================

class RAGSystem:
    """Complete Retrieval-Augmented Generation System"""
    
    def __init__(self, 
                pdf_path: str,
                embedding_model: str = "all-mpnet-base-v2",
                claude_model: str = "claude-3-haiku-20240307"):
        """
        Initialize RAG system
        
        Args:
            pdf_path: Path to Apple PDF
            embedding_model: Embedding model to use
            claude_model: Claude model to use
        """
        self.pdf_path = pdf_path
        self.embedding_model_name = embedding_model
        self.claude_model = claude_model
        
        # Components (initialized on demand)
        self.document_processor = None
        self.embedding_manager = None
        self.retriever = None
        self.generator = None
        self.chunks = None
        self.embeddings = None
    
    def build(self, 
             chunk_size: int = 500,
             chunk_overlap: int = 50,
             use_cached: bool = False):
        """
        Build the complete RAG system
        
        Args:
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            use_cached: Use cached embeddings if available
        """
        print("\n" + "="*70)
        print("BUILDING RAG SYSTEM")
        print("="*70 + "\n")
        
        # Step 1: Load and chunk document
        print("[STEP 1/4] Processing Document")
        print("-" * 70)
        self.document_processor = DocumentProcessor(self.pdf_path)
        self.document_processor.load_pdf()
        self.chunks = self.document_processor.chunk_document(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.document_processor.save_chunks('data/chunks.json')
        
        # Step 2: Create embeddings
        print("\n[STEP 2/4] Creating Embeddings")
        print("-" * 70)
        self.embedding_manager = EmbeddingManager(self.embedding_model_name)
        
        if use_cached and Path('data/embeddings.pkl').exists():
            self.embeddings = EmbeddingManager.load_embeddings('data/embeddings.pkl')
        else:
            self.embeddings = self.embedding_manager.create_embeddings(self.chunks)
            self.embedding_manager.save_embeddings('data/embeddings.pkl')
        
        # Step 3: Initialize retriever
        print("\n[STEP 3/4] Setting up Retriever")
        print("-" * 70)
        self.retriever = Retriever(
            self.chunks,
            self.embeddings,
            self.embedding_manager.model
        )
        print("✓ Retriever ready")
        
        # Step 4: Initialize generator
        print("\n[STEP 4/4] Setting up Generator")
        print("-" * 70)
        self.generator = Generator(self.claude_model)
        print("✓ Generator ready (Claude)")
        
        print("\n" + "="*70)
        print("✓ RAG SYSTEM READY!")
        print("="*70 + "\n")
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        End-to-end RAG query
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
        
        Returns:
            Dictionary with retrieved context and generated answer
        """
        if not all([self.retriever, self.generator]):
            raise ValueError("System not built. Call build() first.")
        
        # Retrieve
        print(f"\n📝 QUERY: {question}\n")
        print("[RETRIEVE] Searching for relevant content...")
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        
        # Prepare context
        context = "\n\n".join([
            f"[Page {r.page}, Similarity: {r.similarity:.2f}]\n{r.text}"
            for r in retrieved
        ])
        
        # Generate
        print("[GENERATE] Creating answer...")
        answer = self.generator.generate(question, context)
        
        # Return result
        result = {
            'question': question,
            'retrieved_chunks': [asdict(r) for r in retrieved],
            'context': context,
            'answer': answer,
            'metadata': {
                'retrieval_count': len(retrieved),
                'embedding_model': self.embedding_model_name,
                'claude_model': self.claude_model
            }
        }
        
        return result
    
    def save_result(self, result: Dict, filepath: str):
        """Save query result to JSON"""
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✓ Saved result to {filepath}")
    
    def batch_query(self, questions: List[str], top_k: int = 3) -> List[Dict]:
        """
        Process multiple queries
        
        Args:
            questions: List of questions
            top_k: Number of chunks per query
        
        Returns:
            List of results
        """
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n[QUERY {i}/{len(questions)}]")
            result = self.query(question, top_k=top_k)
            results.append(result)
        return results


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

def main():
    """Example usage of RAG system"""
    
    # Create data directory
    Path('data').mkdir(exist_ok=True)
    
    # Initialize system
    rag = RAGSystem(
        pdf_path="apple_organization.pdf",
        embedding_model="all-mpnet-base-v2",
        claude_model="claude-3-haiku-20240307"
    )
    
    # Build system
    rag.build(chunk_size=500, chunk_overlap=50, use_cached=False)
    
    # Example queries
    example_queries = [
        "Why did Steve Jobs implement a functional organization?",
        "What are the three key leadership characteristics at Apple?",
        "How many specialist teams were needed for the iPhone portrait mode?",
    ]
    
    # Process queries
    results = rag.batch_query(example_queries, top_k=3)
    
    # Save results
    with open('results/rag_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Processed {len(results)} queries")
    print("✓ Results saved to results/rag_results.json")
    
    return rag, results


if __name__ == "__main__":
    rag, results = main()
    
    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Try asking questions about Apple's organization:")
    print("(Enter 'exit' to quit)\n")
    
    while True:
        question = input("Your question: ").strip()
        if question.lower() == 'exit':
            break
        
        result = rag.query(question)
        print(f"\n{result['answer']}\n")
        print("-" * 70)