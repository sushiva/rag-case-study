---
title: RAG Assistant - Apple Organization
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.28.1"
app_file: app.py
pinned: false
---

# 🚀 RAG Assistant - Apple Organization Analysis

A **Retrieval-Augmented Generation (RAG)** application that answers questions about Apple's organizational structure and innovation processes using multiple LLMs.

## 📋 Features

- 🎯 **Multi-LLM Support**: Ollama (local), OpenAI, Google Gemini, Claude
- 📄 **PDF-based Knowledge Base**: Automatically processes and indexes documents
- 🔍 **Semantic Search**: Fast similarity-based retrieval using FAISS
- 💻 **Beautiful UI**: Clean Streamlit interface
- 🔒 **No API Keys Stored**: Secure API key input via UI
- 📊 **Evaluation Metrics**: Retrieval quality & answer similarity tracking

... rest of your README

## 🎯 What Can It Do?

Ask questions about:
- Apple's organizational structure
- Functional organization benefits
- Leadership model and characteristics
- Innovation processes
- Cross-functional collaboration
- And more!

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.8+
- Ollama (for local LLM) - [Install here](https://ollama.ai)

### Installation
```bash
# Clone repository
git clone <your-repo-url>
cd rag-case-study

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Ollama model (if using local LLM)
ollama pull llama3.2:1b

# Start Ollama
ollama serve
```

### Running the App
```bash
# In a new terminal, from project root
streamlit run app.py
```

Open your browser at `http://localhost:8501`

## 📝 Usage

1. **Select LLM Provider** (sidebar):
   - **Ollama (Local)**: No API key needed
   - **OpenAI**: Paste your API key
   - **Google Gemini**: Paste your API key
   - **Claude**: Paste your API key

2. **Ask a Question**:
   - Type your question about Apple's organization
   - Click "🔍 Search"

3. **Review Results**:
   - See retrieved relevant documents
   - Read the AI-generated answer
   - Check similarity scores

## 🏗️ Project Structure
```
rag-case-study/
├── app.py                      # Main Streamlit application
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/
│   ├── HBR_How_Apple_Is_Organized_For_Innovation-4.pdf
│   ├── extracted_text.txt
│   ├── chunks.json
│   ├── embeddings.json
│   ├── faiss_index.bin
│   └── metadata.json
│
├── scripts/
│   ├── process_pdf.py          # Extract text from PDF
│   ├── chunk_text.py           # Split text into chunks
│   ├── generate_embeddings.py  # Create embeddings
│   ├── build_vector_db.py      # Build FAISS index
│   ├── evaluate_rag.py         # Evaluate system performance
│   └── test_dataset.json       # Test questions
│
└── utils/
    ├── retriever.py            # Semantic search
    ├── llm_handler.py          # LLM integration
    └── rag_chain.py            # Complete RAG pipeline
```

## 🔧 Configuration

Edit `config.yaml` to customize:
```yaml
# Chunk settings
chunking:
  chunk_size: 500        # Characters per chunk
  chunk_overlap: 50      # Overlap between chunks

# Retrieval settings
retrieval:
  top_k: 5              # Number of results to retrieve

# LLM settings
llm:
  ollama:
    model: "llama3.2:1b"
    temperature: 0.7
```

## 📊 Evaluation

Run comprehensive RAG evaluation:
```bash
python scripts/evaluate_rag.py
```

This generates metrics:
- **Retrieval**: Precision@5, Recall@5, MRR
- **Answer Quality**: Semantic similarity scores
- **Report**: Saved as `evaluation_report.json`

## 🎓 How It Works

### RAG Pipeline:

1. **Document Processing**
   - Extract text from PDF
   - Split into overlapping chunks
   - Generate embeddings using `sentence-transformers`

2. **Vector Storage**
   - Store embeddings in FAISS index
   - Fast similarity search

3. **Query Processing**
   - User asks a question
   - Convert query to embedding
   - Search FAISS for similar chunks (top-5)

4. **Answer Generation**
   - Send retrieved chunks + question to LLM
   - LLM generates contextual answer
   - Return answer with source attribution

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **RAG Framework** | LangChain |
| **Embeddings** | Sentence-Transformers |
| **Vector DB** | FAISS |
| **LLMs** | Ollama, OpenAI, Gemini, Claude |
| **Config** | YAML |

## 🔐 Security

- ✅ API keys **not stored** in code
- ✅ API keys **not committed** to git
- ✅ Secure input masking in UI
- ✅ `.gitignore` excludes sensitive files

## 📈 Next Steps (Version 2)

- [ ] Multiple PDF support
- [ ] Query history & favorites
- [ ] User feedback mechanism
- [ ] Answer source attribution
- [ ] Advanced controls (top_k slider, temperature)
- [ ] Dashboard with analytics

## 🤝 Contributing

Feedback and suggestions welcome! This is Version 1 (Pilot).

## 📄 License

MIT License

## 👨‍💻 Author

[Your Name]

## 📧 Support

For issues or questions, please open an issue on GitHub.

---

**Made with ❤️ for learning RAG systems**