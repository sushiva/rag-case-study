"""
Main orchestrator for Apple RAG Case Study
"""

import json
from pathlib import Path

from src.rag_system import RAGSystem
from src.question_generator import QuestionGenerator, RetrieverEvaluator


def main():
    """Run complete RAG evaluation pipeline"""
    
    print("\n" + "="*70)
    print("RAG CASE STUDY - APPLE ORGANIZATIONAL MODEL")
    print("="*70 + "\n")
    
    # Paths
    pdf_path = "data/HBR_How_Apple_Is_Organized_For_Innovation-4.pdf"
    golden_path = "golden_data/golden_dataset_apple_organization.json"
    chunks_path = "data/chunks.json"
    
    # Verify files
    print("[CHECK] Verifying files...")
    for path in [pdf_path, golden_path]:
        if not Path(path).exists():
            print(f"✗ Missing: {path}")
            return
        print(f"✓ {path}")
    
    Path('results').mkdir(exist_ok=True)
    print("✓ results/ folder ready\n")
    
    # ===== STEP 1: Build RAG System =====
    print("="*70)
    print("STEP 1: Building RAG System")
    print("="*70 + "\n")
    
    rag = RAGSystem(
        pdf_path=pdf_path,
        embedding_model="all-mpnet-base-v2",
        claude_model="claude-3-haiku-20240307"
    )
    
    rag.build(chunk_size=500, chunk_overlap=50, use_cached=True)
    
    # ===== STEP 2: Test RAG =====
    print("\n" + "="*70)
    print("STEP 2: Testing RAG System")
    print("="*70 + "\n")
    
    test_query = "Why did Steve Jobs implement a functional organization?"
    print(f"Query: {test_query}\n")
    
    result = rag.query(test_query)
    print(f"Answer:\n{result['answer']}\n")
    
    # ===== STEP 3: Load Golden Dataset =====
    print("\n" + "="*70)
    print("STEP 3: Loading Golden Dataset")
    print("="*70 + "\n")
    
    with open(golden_path, 'r') as f:
        golden_data = json.load(f)
    
    golden_queries = golden_data['queries']
    print(f"✓ Loaded {len(golden_queries)} expert-curated queries\n")
    
    # ===== STEP 4: Auto-Generate Questions =====
    print("="*70)
    print("STEP 4: Auto-Generating Questions from Chunks")
    print("="*70 + "\n")
    
    chunks = rag.document_processor.load_chunks(chunks_path)
    print(f"✓ Loaded {len(chunks)} chunks\n")
    
    # Convert to dict format - USE ALL CHUNKS (no [:5] limit!)
    chunk_dicts = [
        {'id': c.id, 'text': c.text, 'page': c.page}
        for c in chunks  # NO LIMIT - USE ALL!
    ]
    
    print(f"Generating questions for {len(chunk_dicts)} chunks...\n")
    
    generator = QuestionGenerator(model="claude-3-haiku-20240307")
    questions = generator.generate_bulk(
        chunks=chunk_dicts,
        questions_per_chunk=3  # 3 questions per chunk
    )
    
    generator.print_statistics()
    generator.export_to_json('results/auto_generated_questions.json')
    
    # ===== STEP 5: Evaluate Retriever =====
    print("\n" + "="*70)
    print("STEP 5: Evaluating Retriever on Auto-Generated Questions")
    print("="*70 + "\n")
    
    evaluator = RetrieverEvaluator(rag.retriever, questions)
    eval_results = evaluator.evaluate(top_k=3)
    
    with open('results/auto_evaluation.json', 'w') as f:
        json.dump(eval_results, f, indent=2, default=str)
    
    # ===== FINAL SUMMARY =====
    print("\n" + "="*70)
    print("✓ EVALUATION PIPELINE COMPLETE!")
    print("="*70)
    
    print("\n📊 RESULTS SUMMARY:")
    print(f"  • Golden Dataset Queries: {len(golden_queries)}")
    print(f"  • Auto-Generated Questions: {len(questions)}")
    print(f"  • Retrieval Accuracy: {eval_results['summary']['accuracy']:.1%}")
    print(f"  • Mean Reciprocal Rank: {eval_results['summary']['mean_reciprocal_rank']:.2f}")
    
    print("\n📁 OUTPUT FILES:")
    print("  ✓ results/auto_generated_questions.json")
    print("  ✓ results/auto_evaluation.json")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
