"""
Create visualizations for case study results
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
auto_eval = json.load(open('results/auto_evaluation.json'))
golden_eval = json.load(open('results/golden_dataset_4metric_evaluation.json'))

# Create visualizations directory
import os
os.makedirs('results/visualizations', exist_ok=True)

# 1. Retrieval Accuracy by Complexity
fig, ax = plt.subplots(figsize=(10, 6))
complexities = list(auto_eval['by_complexity'].keys())
accuracies = [auto_eval['by_complexity'][c] * 100 for c in complexities]
colors = ['#2ecc71', '#3498db', '#e74c3c']
ax.bar(complexities, accuracies, color=colors)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xlabel('Question Complexity', fontsize=12)
ax.set_title('Retrieval Accuracy by Question Complexity', fontsize=14, fontweight='bold')
ax.set_ylim([0, 100])
for i, v in enumerate(accuracies):
    ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('results/visualizations/01_retrieval_by_complexity.png', dpi=300)
print("✓ Saved: 01_retrieval_by_complexity.png")
plt.close()

# 2. 4-Metric Evaluation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
metrics = ['Similarity', 'Relevance', 'Coherence', 'Groundedness']
values = [
    golden_eval['summary']['metrics']['similarity']['mean'],
    golden_eval['summary']['metrics']['relevance']['mean'] / 5,
    golden_eval['summary']['metrics']['coherence']['mean'] / 5,
    golden_eval['summary']['metrics']['groundedness']['mean'] / 5,
]
colors_metrics = ['#2ecc71' if v > 0.7 else '#f39c12' if v > 0.6 else '#e74c3c' for v in values]
bars = ax.barh(metrics, values, color=colors_metrics)
ax.set_xlim([0, 1])
ax.set_xlabel('Score (0-1 normalized)', fontsize=12)
ax.set_title('Generation Quality - 4 Metrics', fontsize=14, fontweight='bold')
for i, (bar, v) in enumerate(zip(bars, values)):
    ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('results/visualizations/02_4metrics_evaluation.png', dpi=300)
print("✓ Saved: 02_4metrics_evaluation.png")
plt.close()

# 3. Overall System Performance
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['Retrieval\nAccuracy', 'Semantic\nSimilarity', 'Overall\nQuality']
scores = [
    auto_eval['summary']['accuracy'],
    golden_eval['summary']['metrics']['similarity']['mean'],
    golden_eval['summary']['overall_score'],
]
colors_perf = ['#2ecc71', '#3498db', '#f39c12']
bars = ax.bar(categories, scores, color=colors_perf, width=0.6)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('System Performance Overview', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1])
for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{score:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('results/visualizations/03_system_performance.png', dpi=300)
print("✓ Saved: 03_system_performance.png")
plt.close()

# 4. Scale Testing Results
fig, ax = plt.subplots(figsize=(10, 6))
test_types = ['Auto-Generated\nQuestions', 'Golden\nQueries', 'Chunks\nTested']
counts = [
    auto_eval['summary']['total_questions'],
    len(golden_eval['results']),
    int(auto_eval['summary']['chunk_coverage'] * 50),  # Approximate chunks
]
ax.bar(test_types, counts, color=['#3498db', '#2ecc71', '#9b59b6'], width=0.6)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Scale Testing Coverage', fontsize=14, fontweight='bold')
for i, v in enumerate(counts):
    ax.text(i, v + 5, str(v), ha='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('results/visualizations/04_scale_testing.png', dpi=300)
print("✓ Saved: 04_scale_testing.png")
plt.close()

# 5. Metric Distribution (Similarity scores)
fig, ax = plt.subplots(figsize=(10, 6))
similarities = [r['similarity'] for r in golden_eval['results']]
ax.hist(similarities, bins=10, color='#3498db', edgecolor='black', alpha=0.7)
ax.axvline(np.mean(similarities), color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {np.mean(similarities):.3f}')
ax.set_xlabel('Similarity Score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Semantic Similarity Scores', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('results/visualizations/05_similarity_distribution.png', dpi=300)
print("✓ Saved: 05_similarity_distribution.png")
plt.close()

print("\n✓ All visualizations created!")
print("✓ Saved to: results/visualizations/")
