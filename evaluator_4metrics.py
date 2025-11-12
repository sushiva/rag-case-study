"""
4-Metric Evaluator for Golden Dataset
Evaluates: Similarity, Relevance, Coherence, Groundedness
"""

import json
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer, util
import sys

sys.path.insert(0, str('src'))
from rag_system import RAGSystem


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a single query"""
    query_id: int
    query: str
    expert_answer: str
    generated_answer: str
    similarity: float
    relevance: int
    coherence: int
    groundedness: int
    hallucinations: str
    overall_score: float


class GoldenDatasetEvaluator:
    """Evaluate RAG system on golden dataset with 4 metrics"""
    
    def __init__(self, rag_system: RAGSystem, golden_queries: List[Dict]):
        """
        Initialize evaluator
        
        Args:
            rag_system: RAG system to evaluate
            golden_queries: List of golden query dicts
        """
        self.rag = rag_system
        self.queries = golden_queries
        self.client = Anthropic()
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        self.results = []
    
    def evaluate_similarity(self, expert_answer: str, generated_answer: str) -> float:
        """
        Calculate semantic similarity between answers (0-1)
        
        Args:
            expert_answer: Reference answer from expert
            generated_answer: Generated answer from RAG system
        
        Returns:
            Similarity score 0-1
        """
        expert_emb = self.embedding_model.encode(expert_answer, convert_to_tensor=True)
        generated_emb = self.embedding_model.encode(generated_answer, convert_to_tensor=True)
        
        similarity = util.pytorch_cos_sim(expert_emb, generated_emb)
        return float(similarity[0][0])
    
    def evaluate_relevance(self, query: str, answer: str) -> int:
        """
        Evaluate relevance of answer to query (1-5)
        
        Uses Claude as judge
        
        Args:
            query: User query
            answer: Generated answer
        
        Returns:
            Relevance score 1-5
        """
        prompt = f"""Evaluate how well this answer addresses the query on a scale of 1-5.

Query: {query}

Answer: {answer}

Rating scale:
5 = Perfectly answers the query
4 = Mostly answers the query
3 = Partially answers the query
2 = Tangentially related
1 = Not relevant

Return ONLY the number (1-5), nothing else."""
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            
            score = int(response.content[0].text.strip())
            return max(1, min(5, score))  # Ensure 1-5 range
        except:
            return 3  # Default middle score if error
    
    def evaluate_coherence(self, answer: str) -> int:
        """
        Evaluate coherence of answer (1-5)
        
        Uses Claude as judge
        
        Args:
            answer: Generated answer
        
        Returns:
            Coherence score 1-5
        """
        prompt = f"""Evaluate the coherence and clarity of this answer on a scale of 1-5.

Answer: {answer}

Rating scale:
5 = Well-structured, clear, and logical
4 = Generally coherent with minor issues
3 = Somewhat disjointed
2 = Lacks clear structure
1 = Incoherent

Return ONLY the number (1-5), nothing else."""
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            
            score = int(response.content[0].text.strip())
            return max(1, min(5, score))
        except:
            return 3
    
    def evaluate_groundedness(self, answer: str, context: str) -> Tuple[int, str]:
        """
        Evaluate groundedness of answer in context (1-5)
        
        Detects hallucinations
        
        Uses Claude as judge
        
        Args:
            answer: Generated answer
            context: Retrieved context from RAG
        
        Returns:
            Tuple of (groundedness_score, hallucinations_detected)
        """
        prompt = f"""Evaluate how grounded this answer is in the provided context.

Context (max 500 chars):
{context[:500]}

Answer:
{answer}

Groundedness scale:
5 = Fully grounded in context
4 = Mostly grounded (90%+)
3 = Partially grounded (60-89%)
2 = Poorly grounded (<60%)
1 = Mostly hallucinated

Also identify any hallucinations (claims not supported by context).

Format your response as:
SCORE: [1-5]
HALLUCINATIONS: [list any hallucinations or "None"]"""
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.content[0].text
            
            # Parse score
            score_line = [l for l in text.split('\n') if 'SCORE:' in l]
            score = 3
            if score_line:
                try:
                    score = int(score_line[0].split(':')[1].strip())
                    score = max(1, min(5, score))
                except:
                    pass
            
            # Parse hallucinations
            hall_line = [l for l in text.split('\n') if 'HALLUCINATIONS:' in l]
            hallucinations = "None"
            if hall_line:
                hallucinations = hall_line[0].split(':')[1].strip()
            
            return score, hallucinations
        except:
            return 3, "Unable to evaluate"
    
    def evaluate_query(self, query_item: Dict) -> EvaluationMetrics:
        """
        Evaluate a single golden query with all 4 metrics
        
        Args:
            query_item: Golden query dict
        
        Returns:
            EvaluationMetrics object
        """
        query_id = query_item['id']
        query = query_item['query']
        expert_answer = query_item['expert_answer']
        
        print(f"\n[Query {query_id}] {query[:60]}...")
        
        # Run RAG pipeline
        result = self.rag.query(query, top_k=3)
        generated_answer = result['answer']
        context = result['context']
        
        print(f"  → Calculating similarity...")
        similarity = self.evaluate_similarity(expert_answer, generated_answer)
        
        print(f"  → Evaluating relevance...")
        relevance = self.evaluate_relevance(query, generated_answer)
        
        print(f"  → Evaluating coherence...")
        coherence = self.evaluate_coherence(generated_answer)
        
        print(f"  → Evaluating groundedness...")
        groundedness, hallucinations = self.evaluate_groundedness(generated_answer, context)
        
        # Calculate overall score (average of normalized metrics)
        overall_score = (
            similarity +
            (relevance / 5) +
            (coherence / 5) +
            (groundedness / 5)
        ) / 4
        
        metrics = EvaluationMetrics(
            query_id=query_id,
            query=query,
            expert_answer=expert_answer,
            generated_answer=generated_answer,
            similarity=round(similarity, 3),
            relevance=relevance,
            coherence=coherence,
            groundedness=groundedness,
            hallucinations=hallucinations,
            overall_score=round(overall_score, 3)
        )
        
        print(f"  ✓ Similarity: {similarity:.2f}, Relevance: {relevance}/5, "
              f"Coherence: {coherence}/5, Groundedness: {groundedness}/5")
        
        return metrics
    
    def evaluate_all(self) -> Dict:
        """
        Evaluate all golden queries
        
        Returns:
            Comprehensive evaluation report
        """
        print("\n" + "="*70)
        print("4-METRIC EVALUATION - GOLDEN DATASET")
        print("="*70)
        print(f"Evaluating {len(self.queries)} queries...\n")
        
        results = []
        similarities = []
        relevances = []
        coherences = []
        groundedness_scores = []
        
        for i, query in enumerate(self.queries, 1):
            print(f"\n[{i}/{len(self.queries)}]", end="")
            
            metrics = self.evaluate_query(query)
            results.append(metrics)
            
            similarities.append(metrics.similarity)
            relevances.append(metrics.relevance)
            coherences.append(metrics.coherence)
            groundedness_scores.append(metrics.groundedness)
        
        # Calculate summary statistics
        import numpy as np
        
        summary = {
            'total_queries': len(self.queries),
            'metrics': {
                'similarity': {
                    'mean': round(np.mean(similarities), 3),
                    'min': round(np.min(similarities), 3),
                    'max': round(np.max(similarities), 3),
                },
                'relevance': {
                    'mean': round(np.mean(relevances), 2),
                    'min': int(np.min(relevances)),
                    'max': int(np.max(relevances)),
                },
                'coherence': {
                    'mean': round(np.mean(coherences), 2),
                    'min': int(np.min(coherences)),
                    'max': int(np.max(coherences)),
                },
                'groundedness': {
                    'mean': round(np.mean(groundedness_scores), 2),
                    'min': int(np.min(groundedness_scores)),
                    'max': int(np.max(groundedness_scores)),
                }
            },
            'overall_score': round(np.mean([
                np.mean(similarities),
                np.mean(relevances) / 5,
                np.mean(coherences) / 5,
                np.mean(groundedness_scores) / 5
            ]), 3)
        }
        
        report = {
            'summary': summary,
            'results': [
                {
                    'query_id': m.query_id,
                    'query': m.query,
                    'expert_answer': m.expert_answer,
                    'generated_answer': m.generated_answer,
                    'similarity': m.similarity,
                    'relevance': m.relevance,
                    'coherence': m.coherence,
                    'groundedness': m.groundedness,
                    'hallucinations': m.hallucinations,
                    'overall_score': m.overall_score
                }
                for m in results
            ]
        }
        
        self._print_report(report)
        return report
    
    def _print_report(self, report: Dict):
        """Print formatted report"""
        
        print("\n\n" + "="*70)
        print("4-METRIC EVALUATION REPORT")
        print("="*70 + "\n")
        
        summary = report['summary']
        
        print(f"Total Queries Evaluated: {summary['total_queries']}\n")
        
        print("METRICS SUMMARY:")
        print("-" * 70)
        
        print(f"Similarity (0-1):")
        print(f"  Mean: {summary['metrics']['similarity']['mean']}")
        print(f"  Range: {summary['metrics']['similarity']['min']} - {summary['metrics']['similarity']['max']}\n")
        
        print(f"Relevance (1-5):")
        print(f"  Mean: {summary['metrics']['relevance']['mean']:.1f}")
        print(f"  Range: {summary['metrics']['relevance']['min']} - {summary['metrics']['relevance']['max']}\n")
        
        print(f"Coherence (1-5):")
        print(f"  Mean: {summary['metrics']['coherence']['mean']:.1f}")
        print(f"  Range: {summary['metrics']['coherence']['min']} - {summary['metrics']['coherence']['max']}\n")
        
        print(f"Groundedness (1-5):")
        print(f"  Mean: {summary['metrics']['groundedness']['mean']:.1f}")
        print(f"  Range: {summary['metrics']['groundedness']['min']} - {summary['metrics']['groundedness']['max']}\n")
        
        print(f"OVERALL SCORE: {summary['overall_score']:.3f}/1.0")
        
        quality = "EXCELLENT" if summary['overall_score'] > 0.80 else \
                  "GOOD" if summary['overall_score'] > 0.70 else \
                  "FAIR" if summary['overall_score'] > 0.60 else "NEEDS IMPROVEMENT"
        
        print(f"Quality Assessment: {quality}\n")
        print("="*70 + "\n")


def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print("GOLDEN DATASET 4-METRIC EVALUATION")
    print("Apple Organizational Model Case Study")
    print("="*70)
    
    # Paths
    pdf_path = "data/HBR_How_Apple_Is_Organized_For_Innovation-4.pdf"
    golden_path = "golden_data/golden_dataset_apple_organization.json"
    
    # Load RAG system
    print("\n[STEP 1] Initializing RAG System...")
    rag = RAGSystem(
        pdf_path=pdf_path,
        embedding_model="all-mpnet-base-v2",
        claude_model="claude-3-haiku-20240307"
    )
    rag.build(chunk_size=500, chunk_overlap=50, use_cached=True)
    
    # Load golden dataset
    print("\n[STEP 2] Loading Golden Dataset...")
    with open(golden_path, 'r') as f:
        golden_data = json.load(f)
    golden_queries = golden_data['queries']
    print(f"✓ Loaded {len(golden_queries)} queries")
    
    # Evaluate
    print("\n[STEP 3] Running 4-Metric Evaluation...")
    evaluator = GoldenDatasetEvaluator(rag, golden_queries)
    report = evaluator.evaluate_all()
    
    # Save report
    print("\n[STEP 4] Saving Report...")
    with open('results/golden_dataset_4metric_evaluation.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("✓ Saved to: results/golden_dataset_4metric_evaluation.json")
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()