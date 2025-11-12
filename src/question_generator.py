"""
Complete Auto-Question Generation System
Apple Organizational Model Case Study

This generates diverse questions at different complexity levels
from document chunks using Claude API.

Ready to use with your RAG system.
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import time

# Dependencies:
# pip install anthropic

from anthropic import Anthropic


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class GeneratedQuestion:
    """Auto-generated question with metadata"""
    question_id: int
    question: str
    chunk_id: int
    chunk_text: str
    source_page: int
    complexity: str  # 'basic', 'intermediate', 'advanced'
    generation_model: str = "claude-3-haiku-20240307"


# ============================================================================
# QUESTION GENERATION TEMPLATES & PROMPTS
# ============================================================================

class QuestionTemplates:
    """Collection of prompts for generating questions at different levels"""
    
    @staticmethod
    def basic_prompt(chunk_text: str) -> str:
        """Generate a basic factual question"""
        return f"""Generate a simple, factual question about this text that can be answered directly from it.

TEXT:
{chunk_text[:500]}

The question should:
- Be answerable directly from the text
- Test factual recall
- Be phrased naturally
- Be 1-2 sentences

IMPORTANT: Return ONLY the question text, nothing else."""
    
    @staticmethod
    def intermediate_prompt(chunk_text: str) -> str:
        """Generate an intermediate complexity question"""
        return f"""Generate a question that requires understanding and reasoning about this text.

TEXT:
{chunk_text[:500]}

The question should:
- Require understanding relationships and context
- Not be answerable by simple keyword matching
- Test comprehension
- Be phrased naturally
- Be 1-3 sentences

IMPORTANT: Return ONLY the question text, nothing else."""
    
    @staticmethod
    def advanced_prompt(chunk_text: str) -> str:
        """Generate an advanced critical thinking question"""
        return f"""Generate a challenging question that requires analysis, inference, or critical thinking about this text.

TEXT:
{chunk_text[:500]}

The question should:
- Require deeper analysis or synthesis
- Go beyond surface-level understanding
- Encourage critical thinking
- Be relevant to business/organizational context
- Be phrased naturally
- Be 1-3 sentences

IMPORTANT: Return ONLY the question text, nothing else."""
    
    @staticmethod
    def why_prompt(chunk_text: str) -> str:
        """Generate a 'why' question about causality/motivation"""
        return f"""Generate a "Why" question about this text that explores causality, motivation, or reasoning.

TEXT:
{chunk_text[:500]}

The question should:
- Start with "Why" or "How" or "What is the reason"
- Explore underlying causes or reasoning
- Be answerable from the text
- Be naturally phrased

IMPORTANT: Return ONLY the question text, nothing else."""
    
    @staticmethod
    def comparison_prompt(chunk_text: str) -> str:
        """Generate a comparison/contrast question"""
        return f"""Generate a comparison or contrast question about this text.

TEXT:
{chunk_text[:500]}

The question should:
- Ask about similarities, differences, or relationships between concepts
- Start with phrases like "How is X different from Y?" or "What is the relationship between..."
- Be answerable from the text
- Be naturally phrased

IMPORTANT: Return ONLY the question text, nothing else."""


# ============================================================================
# QUESTION GENERATOR
# ============================================================================

class QuestionGenerator:
    """Generate evaluation questions from document chunks using Claude"""
    
    def __init__(self, model: str = "claude-3-haiku-20240307"):
        """
        Initialize question generator
        
        Args:
            model: Claude model to use
        """
        self.client = Anthropic()
        self.model = model
        self.generated_questions = []
        self.question_counter = 0
        
        print(f"✓ Question Generator initialized (model: {model})")
    
    def _call_claude(self, prompt: str) -> str:
        """
        Call Claude API to generate question
        
        Args:
            prompt: Prompt for Claude
        
        Returns:
            Generated question text
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            question_text = response.content[0].text.strip()
            
            # Clean up the question
            if question_text.startswith('"'):
                question_text = question_text[1:]
            if question_text.endswith('"'):
                question_text = question_text[:-1]
            
            return question_text
            
        except Exception as e:
            print(f"✗ Error calling Claude: {e}")
            raise
    
    def generate_basic_question(self, 
                               chunk_id: int,
                               chunk_text: str,
                               source_page: int) -> GeneratedQuestion:
        """Generate a basic-level question"""
        self.question_counter += 1
        
        print(f"  → Generating basic question...")
        prompt = QuestionTemplates.basic_prompt(chunk_text)
        question_text = self._call_claude(prompt)
        
        return GeneratedQuestion(
            question_id=self.question_counter,
            question=question_text,
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            source_page=source_page,
            complexity='basic',
            generation_model=self.model
        )
    
    def generate_intermediate_question(self,
                                      chunk_id: int,
                                      chunk_text: str,
                                      source_page: int) -> GeneratedQuestion:
        """Generate an intermediate-level question"""
        self.question_counter += 1
        
        print(f"  → Generating intermediate question...")
        prompt = QuestionTemplates.intermediate_prompt(chunk_text)
        question_text = self._call_claude(prompt)
        
        return GeneratedQuestion(
            question_id=self.question_counter,
            question=question_text,
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            source_page=source_page,
            complexity='intermediate',
            generation_model=self.model
        )
    
    def generate_advanced_question(self,
                                  chunk_id: int,
                                  chunk_text: str,
                                  source_page: int) -> GeneratedQuestion:
        """Generate an advanced-level question"""
        self.question_counter += 1
        
        print(f"  → Generating advanced question...")
        prompt = QuestionTemplates.advanced_prompt(chunk_text)
        question_text = self._call_claude(prompt)
        
        return GeneratedQuestion(
            question_id=self.question_counter,
            question=question_text,
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            source_page=source_page,
            complexity='advanced',
            generation_model=self.model
        )
    
    def generate_why_question(self,
                             chunk_id: int,
                             chunk_text: str,
                             source_page: int) -> GeneratedQuestion:
        """Generate a 'why' question about causality"""
        self.question_counter += 1
        
        print(f"  → Generating 'why' question...")
        prompt = QuestionTemplates.why_prompt(chunk_text)
        question_text = self._call_claude(prompt)
        
        return GeneratedQuestion(
            question_id=self.question_counter,
            question=question_text,
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            source_page=source_page,
            complexity='intermediate',
            generation_model=self.model
        )
    
    def generate_comparison_question(self,
                                    chunk_id: int,
                                    chunk_text: str,
                                    source_page: int) -> GeneratedQuestion:
        """Generate a comparison/contrast question"""
        self.question_counter += 1
        
        print(f"  → Generating comparison question...")
        prompt = QuestionTemplates.comparison_prompt(chunk_text)
        question_text = self._call_claude(prompt)
        
        return GeneratedQuestion(
            question_id=self.question_counter,
            question=question_text,
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            source_page=source_page,
            complexity='intermediate',
            generation_model=self.model
        )
    
    def generate_for_chunk(self,
                          chunk_id: int,
                          chunk_text: str,
                          source_page: int,
                          num_questions: int = 5) -> List[GeneratedQuestion]:
        """
        Generate multiple questions for a single chunk
        
        Args:
            chunk_id: Unique identifier for chunk
            chunk_text: The chunk content
            source_page: Source page number
            num_questions: How many questions to generate (1-5)
        
        Returns:
            List of GeneratedQuestion objects
        """
        questions = []
        
        print(f"\n[CHUNK {chunk_id}] Generating {num_questions} questions...")
        
        # Always include basic
        if num_questions >= 1:
            q = self.generate_basic_question(chunk_id, chunk_text, source_page)
            questions.append(q)
            time.sleep(0.5)  # Avoid rate limiting
        
        # Add intermediate
        if num_questions >= 2:
            q = self.generate_intermediate_question(chunk_id, chunk_text, source_page)
            questions.append(q)
            time.sleep(0.5)
        
        # Add advanced
        if num_questions >= 3:
            q = self.generate_advanced_question(chunk_id, chunk_text, source_page)
            questions.append(q)
            time.sleep(0.5)
        
        # Add 'why' question
        if num_questions >= 4:
            q = self.generate_why_question(chunk_id, chunk_text, source_page)
            questions.append(q)
            time.sleep(0.5)
        
        # Add comparison question
        if num_questions >= 5:
            q = self.generate_comparison_question(chunk_id, chunk_text, source_page)
            questions.append(q)
            time.sleep(0.5)
        
        self.generated_questions.extend(questions)
        return questions
    
    def generate_bulk(self,
                     chunks: List[Dict],
                     questions_per_chunk: int = 3) -> List[GeneratedQuestion]:
        """
        Generate questions for all chunks
        
        Args:
            chunks: List of chunks with 'id', 'text', 'page'
            questions_per_chunk: How many questions per chunk
        
        Returns:
            All generated questions
        """
        print("\n" + "="*70)
        print("GENERATING QUESTIONS FROM CHUNKS")
        print("="*70)
        print(f"Total chunks: {len(chunks)}")
        print(f"Questions per chunk: {questions_per_chunk}")
        print(f"Total questions to generate: {len(chunks) * questions_per_chunk}")
        print("="*70 + "\n")
        
        all_questions = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"[PROCESSING {i}/{len(chunks)}]")
            
            questions = self.generate_for_chunk(
                chunk_id=chunk['id'],
                chunk_text=chunk['text'],
                source_page=chunk['page'],
                num_questions=questions_per_chunk
            )
            
            all_questions.extend(questions)
            
            # Show progress
            print(f"✓ Generated {len(questions)} questions")
        
        print(f"\n✓ Total questions generated: {len(all_questions)}")
        return all_questions
    
    def export_to_json(self, filepath: str):
        """
        Export generated questions to JSON
        
        Args:
            filepath: Where to save JSON
        """
        data = {
            'metadata': {
                'total_questions': len(self.generated_questions),
                'generation_model': self.model,
                'source': 'Auto-Generated from Apple PDF chunks',
                'purpose': 'Retriever evaluation and scale testing'
            },
            'questions': [asdict(q) for q in self.generated_questions]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Exported {len(self.generated_questions)} questions to {filepath}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about generated questions"""
        by_complexity = {}
        for q in self.generated_questions:
            complexity = q.complexity
            by_complexity[complexity] = by_complexity.get(complexity, 0) + 1
        
        return {
            'total_questions': len(self.generated_questions),
            'by_complexity': by_complexity,
            'unique_chunks': len(set(q.chunk_id for q in self.generated_questions))
        }
    
    def print_statistics(self):
        """Print statistics to console"""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("QUESTION GENERATION STATISTICS")
        print("="*70)
        print(f"Total Questions Generated: {stats['total_questions']}")
        print(f"Unique Chunks: {stats['unique_chunks']}")
        print("\nBy Complexity:")
        for complexity, count in stats['by_complexity'].items():
            percentage = (count / stats['total_questions']) * 100
            print(f"  • {complexity.capitalize()}: {count} ({percentage:.1f}%)")
        print("="*70 + "\n")


# ============================================================================
# RETRIEVER EVALUATOR FOR AUTO-GENERATED QUESTIONS
# ============================================================================

class RetrieverEvaluator:
    """Evaluate retriever performance on auto-generated questions"""
    
    def __init__(self, retriever, auto_questions: List[GeneratedQuestion]):
        """
        Args:
            retriever: RAG retriever to test
            auto_questions: List of GeneratedQuestion objects
        """
        self.retriever = retriever
        self.questions = auto_questions
        self.results = []
    
    def evaluate(self, top_k: int = 3) -> Dict:
        """
        Evaluate retriever on all auto-generated questions
        
        Metrics:
        - Accuracy: % where correct chunk is in top-k
        - MRR: Mean Reciprocal Rank
        - Coverage: % of chunks tested
        - By complexity: Performance by question type
        """
        
        print("\n" + "="*70)
        print("RETRIEVER EVALUATION - AUTO-GENERATED QUESTIONS")
        print("="*70)
        print(f"Total queries: {len(self.questions)}")
        print(f"Top-k: {top_k}\n")
        
        correct = 0
        reciprocal_ranks = []
        covered_chunks = set()
        by_complexity = {'basic': [], 'intermediate': [], 'advanced': []}
        detailed_results = []
        
        for i, q in enumerate(self.questions, 1):
            # Run retrieval
            retrieved = self.retriever.retrieve(q.question, top_k=top_k)
            retrieved_ids = [r.chunk_id for r in retrieved]
            
            # Check if correct chunk retrieved
            is_correct = q.chunk_id in retrieved_ids
            if is_correct:
                correct += 1
            
            # Calculate rank
            try:
                rank = retrieved_ids.index(q.chunk_id) + 1
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                reciprocal_ranks.append(0)
            
            # Track coverage
            covered_chunks.add(q.chunk_id)
            
            # Track by complexity
            by_complexity[q.complexity].append(is_correct)
            
            # Store detailed result
            detailed_results.append({
                'question_id': q.question_id,
                'question': q.question,
                'expected_chunk': q.chunk_id,
                'retrieved_chunks': retrieved_ids,
                'correct': is_correct,
                'complexity': q.complexity
            })
            
            # Progress
            if (i % 20 == 0) or (i == len(self.questions)):
                accuracy = correct / i
                print(f"Progress: {i}/{len(self.questions)} | Accuracy so far: {accuracy:.1%}")
        
        # Calculate metrics
        total = len(self.questions)
        accuracy = correct / total if total > 0 else 0
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
        total_chunks = len(set(q.chunk_id for q in self.questions))
        coverage = len(covered_chunks) / total_chunks if total_chunks > 0 else 0
        
        # By complexity
        complexity_scores = {}
        for level, results in by_complexity.items():
            if results:
                complexity_scores[level] = sum(results) / len(results)
        
        report = {
            'summary': {
                'total_questions': total,
                'correct_retrievals': correct,
                'accuracy': round(accuracy, 3),
                'mean_reciprocal_rank': round(mrr, 3),
                'chunk_coverage': round(coverage, 3),
            },
            'by_complexity': {k: round(v, 3) for k, v in complexity_scores.items()},
            'detailed_results': detailed_results
        }
        
        self._print_report(report)
        return report
    
    def _print_report(self, report: Dict):
        """Print evaluation report"""
        
        print("\n" + "="*70)
        print("AUTO-GENERATED QUESTION EVALUATION RESULTS")
        print("="*70 + "\n")
        
        summary = report['summary']
        print(f"Total Questions Evaluated:  {summary['total_questions']}")
        print(f"Correct Retrievals:         {summary['correct_retrievals']}/{summary['total_questions']}")
        print(f"\nMETRICS:")
        print(f"  • Accuracy:               {summary['accuracy']:.1%}")
        print(f"  • Mean Reciprocal Rank:   {summary['mean_reciprocal_rank']:.2f}")
        print(f"  • Chunk Coverage:         {summary['chunk_coverage']:.1%}")
        
        if report['by_complexity']:
            print(f"\nPERFORMANCE BY COMPLEXITY:")
            for level, score in report['by_complexity'].items():
                print(f"  • {level.capitalize()}: {score:.1%}")
        
        print("\n" + "="*70 + "\n")
        
        return report


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

def main():
    """Example usage of question generator"""
    
    print("\n" + "="*70)
    print("AUTO-QUESTION GENERATION SYSTEM")
    print("For Apple Organizational Model Case Study")
    print("="*70 + "\n")
    
    # Create output directory
    Path('data').mkdir(exist_ok=True)
    
    # Example chunks (in practice, load from your document processor)
    example_chunks = [
        {
            'id': 1,
            'page': 4,
            'text': 'When Jobs arrived back at Apple, it had a conventional structure for a company of its size and scope. It was divided into business units, each with its own P&L responsibilities. General managers ran the Macintosh products group, the information appliances division, and the server products division, among others. Believing that conventional management had stifled innovation, Jobs, in his first year returning as CEO, laid off the general managers of all the business units (in a single day), put the entire company under one P&L, and combined the disparate functional departments of the business units into one functional organization.'
        },
        {
            'id': 2,
            'page': 4,
            'text': 'What is surprising—in fact, remarkable—is that Apple retains it today, even though the company is nearly 40 times as large in terms of revenue and far more complex than it was in 1998. Senior vice presidents are in charge of functions, not products. As was the case with Jobs before him, CEO Tim Cook occupies the only position on the organizational chart where the design, engineering, operations, marketing, and retail of any of Apple\'s main products meet.'
        },
        {
            'id': 3,
            'page': 6,
            'text': 'Apple is not a company where general managers oversee managers; rather, it is a company where experts lead experts. The assumption is that it\'s easier to train an expert to manage well than to train a manager to be an expert. At Apple, hardware experts manage hardware, software experts software, and so on. This approach cascades down all levels of the organization through areas of ever-increasing specialization.'
        },
    ]
    
    # Initialize generator
    generator = QuestionGenerator(
        model="claude-3-haiku-20240307"
    )
    
    # Generate questions (3 per chunk for demo, use more in production)
    print("[STEP 1] Generating Questions from Chunks")
    print("-" * 70)
    questions = generator.generate_bulk(
        chunks=example_chunks,
        questions_per_chunk=3
    )
    
    # Print statistics
    print("\n[STEP 2] Generation Statistics")
    print("-" * 70)
    generator.print_statistics()
    
    # Export to JSON
    print("[STEP 3] Exporting Questions")
    print("-" * 70)
    generator.export_to_json('data/auto_generated_questions.json')
    
    # Show sample questions
    print("\n[STEP 4] Sample Generated Questions")
    print("-" * 70)
    for q in questions[:6]:
        print(f"\nQ{q.question_id}: {q.question}")
        print(f"   [Chunk {q.chunk_id}, {q.complexity}, Page {q.source_page}]")
    
    return generator, questions


if __name__ == "__main__":
    generator, questions = main()
    
    print("\n" + "="*70)
    print("HOW TO USE WITH YOUR RAG SYSTEM")
    print("="*70 + """

1. LOAD YOUR CHUNKS:
   from rag_system_complete_example import DocumentProcessor
   processor = DocumentProcessor("apple_organization.pdf")
   processor.load_pdf()
   chunks = processor.chunk_document()

2. GENERATE QUESTIONS:
   from question_generator_complete_example import QuestionGenerator
   generator = QuestionGenerator()
   questions = generator.generate_bulk(chunks, questions_per_chunk=3)

3. EVALUATE RETRIEVER:
   from question_generator_complete_example import RetrieverEvaluator
   from rag_system_complete_example import RAGSystem
   
   rag = RAGSystem("apple_organization.pdf")
   rag.build()
   
   evaluator = RetrieverEvaluator(rag.retriever, questions)
   report = evaluator.evaluate(top_k=3)

4. SAVE RESULTS:
   with open('results/auto_evaluation_report.json', 'w') as f:
       json.dump(report, f, indent=2)

EXPECTED RESULTS:
✓ Generate 200-500 diverse questions
✓ Evaluate retriever at scale
✓ Get retrieval accuracy metrics
✓ Identify weak retrieval cases
""")