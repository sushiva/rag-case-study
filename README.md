# 🎯 RAG Case Study - Apple Organizational Model

A comprehensive **Retrieval-Augmented Generation (RAG)** system with multi-model LLM support for analyzing Apple's organizational structure.

## ✨ Features

- **Multi-Model Support**: Claude, Google Gemini, OpenAI GPT
- **High Accuracy**: 77.9% retrieval accuracy on 249 auto-generated questions
- **Quality Evaluation**: 4-metric framework on 15 expert-curated queries
- **Interactive Dashboard**: Real-time chatbot interface
- **Production Ready**: Secure API key management, comprehensive logging

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Conda or venv
- API keys (Claude, Gemini, or OpenAI)

### Installation
```bash
git clone https://github.com/sushiva/rag-case-study.git
cd rag-case-study

pip install -r requirements.txt
```

### Usage
```bash
streamlit run app.py
```

Then:
1. Select your preferred LLM provider
2. Enter your API key from that provider
3. Ask questions about Apple's organizational model!

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| Retrieval Accuracy | 77.9% |
| Semantic Similarity | 0.787 |
| Overall Quality | 0.647/1.0 |
| Questions Tested | 264 |
| Complexity Levels | Basic, Intermediate, Advanced |

## 🎯 Supported Models

### Claude (Anthropic)
- claude-3-5-sonnet-20241022
- claude-3-5-haiku-20241022
- claude-3-opus-20250219

### Google Gemini
- gemini-2.0-flash
- gemini-1.5-pro
- gemini-1.5-flash

### OpenAI
- gpt-4-turbo
- gpt-4o
- gpt-4o-mini
- gpt-3.5-turbo

## 📁 Project Structure
```
rag-case-study/
├── app.py                          # Main Streamlit app
├── main.py                         # RAG pipeline
├── evaluator_4metrics.py           # Quality evaluation
├── src/
│   ├── rag_system.py              # Core RAG system
│   ├── rag_system_multimodel.py   # Multi-model support
│   └── question_generator.py      # Auto-generation
├── results/
│   ├── auto_evaluation.json       # 249 questions results
│   └── golden_dataset_4metric_evaluation.json
├── data/
│   ├── chunks.json
│   └── embeddings.pkl
└── requirements.txt
```

## 🏗️ System Architecture

1. **Document Processing**
   - PDF loading and chunking (500 tokens, 50-token overlap)
   - ~50 document chunks from Apple case study

2. **Embedding & Retrieval**
   - Sentence-transformers (all-mpnet-base-v2)
   - Semantic similarity search
   - Top-k retrieval (k=3)

3. **Generation (Multi-Model)**
   - Provider: Claude, Gemini, or OpenAI
   - Context-grounded answer generation
   - Consistent interface across all providers

4. **Evaluation**
   - Auto-generated questions: Scale testing
   - Golden dataset: Quality assessment
   - Metrics: Similarity, Relevance, Coherence, Groundedness

## 🔐 Getting API Keys

### Claude (Anthropic)
- Visit: https://console.anthropic.com/
- Create API Key
- Free tier available

### Google Gemini
- Visit: https://aistudio.google.com/
- Get API Key
- Free tier available

### OpenAI
- Visit: https://platform.openai.com/
- Create API Key
- Requires payment setup

## 📖 Usage Examples

### Example 1: Ask with Claude
```
1. Open the app
2. Select "Claude" in sidebar
3. Enter Anthropic API key
4. Ask: "Why did Steve Jobs implement functional organization?"
5. Get instant answer grounded in source document
```

### Example 2: Compare Models
```
1. Ask same question with Claude
2. Switch to Gemini
3. Ask same question
4. Compare answers!
```

## 📊 Dashboard Pages

- **💬 Chatbot**: Real-time query interface
- **📈 Overview**: Key metrics and findings
- **🔍 Retrieval**: Performance analysis
- **💎 Generation**: Quality metrics
- **📋 Analysis**: System comparison
- **📑 Report**: Comprehensive case study
- **ℹ️ About**: Project details

## 🎓 Key Metrics

- **Retrieval**: Accuracy@3, Mean Reciprocal Rank
- **Generation**: 
  - Similarity: Semantic alignment (0-1)
  - Relevance: Answer quality (1-5)
  - Coherence: Structure clarity (1-5)
  - Groundedness: Source grounding (1-5)

## 🔬 Evaluation Approach

**Dual-Track:**
- Auto-generated: 249 diverse questions for scale
- Manual golden: 15 expert-curated queries for quality

**Complexity Levels:**
- Basic: Simple factual questions
- Intermediate: Multi-step reasoning
- Advanced: Complex analysis

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud (Free)
1. Push to GitHub
2. Go to https://streamlit.io/cloud
3. Connect repo
4. Deploy!

## 📝 Citations

- Case Study: HBR - How Apple Is Organized For Innovation
- Embedding Model: Sentence-Transformers (all-mpnet-base-v2)
- LLM Providers: Anthropic, Google, OpenAI

## 📄 License

MIT License - Feel free to use for educational and commercial purposes

## 👤 Author

Shiva - ML/Data Science Portfolio Project

---

**⭐ If you find this project useful, please star it on GitHub!**

