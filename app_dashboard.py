"""
Streamlit Dashboard for RAG Case Study with Interactive Chatbot
Apple Organizational Model - Interactive Results & Query Interface
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add src to path for RAG imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from rag_system import RAGSystem

# Set page config
st.set_page_config(
    page_title="RAG Case Study - Apple",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get current directory
BASE_DIR = Path(__file__).parent.resolve()

# Custom CSS
st.markdown("""
<style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .good { color: #28a745; font-weight: bold; }
    .fair { color: #ffc107; font-weight: bold; }
    .poor { color: #dc3545; font-weight: bold; }
    .context-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-left: 4px solid #3498db;
        border-radius: 5px;
        margin: 10px 0;
    }
    .answer-box {
        background-color: #f0f8e8;
        padding: 15px;
        border-left: 4px solid #2ecc71;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False

# Load data
@st.cache_data
def load_data():
    """Load all evaluation results"""
    try:
        auto_eval_path = BASE_DIR / 'results' / 'auto_evaluation.json'
        golden_eval_path = BASE_DIR / 'results' / 'golden_dataset_4metric_evaluation.json'
        report_path = BASE_DIR / 'COMPREHENSIVE_CASE_STUDY_REPORT.txt'
        
        auto_eval = json.load(open(auto_eval_path))
        golden_eval = json.load(open(golden_eval_path))
        with open(report_path, 'r') as f:
            report = f.read()
        
        return auto_eval, golden_eval, report
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Initialize RAG system
@st.cache_resource
def init_rag_system():
    """Initialize RAG system (cached)"""
    try:
        pdf_path = BASE_DIR / 'data' / 'HBR_How_Apple_Is_Organized_For_Innovation-4.pdf'
        
        rag = RAGSystem(
            pdf_path=str(pdf_path),
            embedding_model="all-mpnet-base-v2",
            claude_model="claude-3-haiku-20240307"
        )
        rag.build(chunk_size=500, chunk_overlap=50, use_cached=True)
        return rag
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None

# Load report content
def load_report():
    """Load comprehensive report"""
    try:
        report_path = BASE_DIR / 'COMPREHENSIVE_CASE_STUDY_REPORT.txt'
        with open(report_path, 'r') as f:
            return f.read()
    except Exception as e:
        st.error(f"Could not load report: {e}")
        return None

# Main app
def main():
    # Header
    st.markdown("# 🎯 RAG Case Study - Apple Organizational Model")
    st.markdown("**Retrieval-Augmented Generation System with Comprehensive Evaluation**")
    
    # Load data
    auto_eval, golden_eval, report = load_data()
    
    if auto_eval is None:
        st.error(f"Could not load evaluation results from {BASE_DIR}")
        st.info("Make sure you're running from the rag-case-study directory:")
        st.code("cd ~/portfolio-project/rag-case-study && streamlit run app_dashboard.py")
        return
    
    # Sidebar navigation
    st.sidebar.markdown("## 📊 Navigation")
    page = st.sidebar.radio(
        "Select Section:",
        [
            "💬 Ask Questions (Chatbot)",
            "📈 Overview",
            "🔍 Retrieval Evaluation",
            "💎 Generation Quality",
            "📋 Combined Analysis",
            "📑 Full Report",
            "ℹ️ About"
        ]
    )
    
    # =====================================================================
    # PAGE 0: CHATBOT
    # =====================================================================
    if page == "💬 Ask Questions (Chatbot)":
        st.markdown("## 🤖 Interactive RAG Chatbot")
        st.markdown("Ask questions about Apple's organizational model and get answers from the document!")
        
        # Initialize RAG system
        with st.spinner("🔄 Initializing RAG system..."):
            rag = init_rag_system()
        
        if rag is None:
            st.error("Failed to initialize RAG system")
            return
        
        st.success("✓ RAG system ready!")
        
        st.markdown("---")
        
        # Example queries
        st.markdown("### 💡 Example Queries")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - Why did Steve Jobs implement a functional organization?
            - What are the key leadership characteristics at Apple?
            - How does Apple's organizational structure enable innovation?
            """)
        
        with col2:
            st.markdown("""
            - What is the role of deep expertise in Apple's success?
            - How many specialist teams were needed for iPhone portrait mode?
            - What is Apple's approach to management?
            """)
        
        st.markdown("---")
        
        # Chat history display
        st.markdown("### 📝 Conversation")
        
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"], avatar="😊" if message["role"] == "user" else "🤖"):
                    st.markdown(message["content"])
                    
                    # Show context if available
                    if message["role"] == "assistant" and "context" in message:
                        with st.expander("📚 View Retrieved Context"):
                            st.markdown(message["context"])
        
        st.markdown("---")
        
        # FIXED VERSION:
        st.markdown("### ✍️ Ask a Question")
        
        user_input = st.chat_input(
            "Ask a question about Apple's organizational model...",
            key="user_input"
        )
        
        # Process user input
        if user_input:
            # Add user message to chat
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Display user message
            with st.chat_message("user", avatar="😊"):
                st.markdown(user_input)
            
            # Generate answer
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🔄 Searching documents and generating answer..."):
                    try:
                        # Query RAG system
                        result = rag.query(user_input, top_k=3)
                        
                        answer = result['answer']
                        context = result['context']
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Show context
                        with st.expander("📚 View Retrieved Context"):
                            st.markdown(context)
                        
                        # Add to chat history with context
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "context": context
                        })
                        
                        # Show source info
                        st.divider()
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Retrieved Chunks", len(result.get('retrieved_results', [])))
                        with col2:
                            if result.get('retrieved_results'):
                                pages = [r.page for r in result['retrieved_results']]
                                st.metric("Source Pages", f"{min(pages)}-{max(pages)}")
                        with col3:
                            st.metric("Model", result['metadata']['claude_model'].split('/')[-1])
                    
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")
        
        # Clear chat button
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # =====================================================================
    # PAGE 1: OVERVIEW
    # =====================================================================
    elif page == "📈 Overview":
        st.markdown("## Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Retrieval Accuracy",
                f"{auto_eval['summary']['accuracy']:.1%}",
                "249 questions"
            )
        
        with col2:
            st.metric(
                "Semantic Similarity",
                f"{golden_eval['summary']['metrics']['similarity']['mean']:.3f}",
                "Expert answers"
            )
        
        with col3:
            st.metric(
                "Overall Quality",
                f"{golden_eval['summary']['overall_score']:.3f}/1.0",
                "4 metrics"
            )
        
        with col4:
            st.metric(
                "Questions Tested",
                f"{auto_eval['summary']['total_questions'] + len(golden_eval['results'])}",
                "Scale tested"
            )
        
        st.markdown("---")
        
        # Key findings
        st.markdown("### 🎯 Key Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Strengths")
            st.markdown(f"""
            - **Strong Retrieval**: {auto_eval['summary']['accuracy']:.1%} accuracy on 249 questions
            - **Good Semantic Alignment**: {golden_eval['summary']['metrics']['similarity']['mean']:.3f} similarity with expert answers
            - **Scale Tested**: System validated at 5x scale
            - **Consistent Performance**: All question types >75% accuracy
            """)
        
        with col2:
            st.markdown("#### ⚠️ Areas for Improvement")
            st.markdown("""
            - Generation quality (Relevance, Coherence): 3/5 (fair)
            - Groundedness and hallucination control
            - Answer structure and clarity
            - Could improve to 0.85+/1.0 with prompt engineering
            """)
    
    # =====================================================================
    # PAGE 2: RETRIEVAL EVALUATION
    # =====================================================================
    elif page == "🔍 Retrieval Evaluation":
        st.markdown("## Retrieval Performance (Scale Testing)")
        
        st.info("""
        This section evaluates the retriever's ability to find relevant document chunks.
        **Metric**: Accuracy@3 (% of queries where correct chunk found in top-3 results)
        """)
        
        # Overall metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Questions", auto_eval['summary']['total_questions'])
        with col2:
            st.metric("Accuracy", f"{auto_eval['summary']['accuracy']:.1%}")
        with col3:
            st.metric("MRR", f"{auto_eval['summary']['mean_reciprocal_rank']:.2f}")
        
        st.markdown("---")
        
        # Performance by complexity
        st.markdown("### Performance by Question Complexity")
        
        complexity_data = auto_eval['by_complexity']
        df_complexity = pd.DataFrame({
            'Complexity': list(complexity_data.keys()),
            'Accuracy': [v * 100 for v in complexity_data.values()]
        })
        
        fig = px.bar(
            df_complexity,
            x='Complexity',
            y='Accuracy',
            title='Retrieval Accuracy by Question Complexity',
            labels={'Accuracy': 'Accuracy (%)'},
            color='Accuracy',
            color_continuous_scale=['#e74c3c', '#f39c12', '#2ecc71'],
            text='Accuracy'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Insights
        st.markdown("### 📊 Insights")
        st.markdown(f"""
        - **Basic Questions**: {complexity_data['basic']*100:.1f}% accuracy - Excellent foundation
        - **Intermediate Questions**: {complexity_data['intermediate']*100:.1f}% accuracy - Strong performance
        - **Advanced Questions**: {complexity_data['advanced']*100:.1f}% accuracy - Good understanding of complex topics
        - **All complexity levels exceed 75% accuracy** - Demonstrates robust semantic search
        """)
    
    # =====================================================================
    # PAGE 3: GENERATION QUALITY
    # =====================================================================
    elif page == "💎 Generation Quality":
        st.markdown("## Generation Quality (4-Metric Evaluation)")
        
        st.info("""
        Evaluated 15 expert-curated queries on 4 metrics:
        - **Similarity**: Semantic alignment with expert answers (0-1)
        - **Relevance**: How well answer addresses query (1-5)
        - **Coherence**: Answer structure and clarity (1-5)
        - **Groundedness**: How grounded in source documents (1-5)
        """)
        
        # Metrics table
        metrics_summary = {
            'Metric': ['Similarity', 'Relevance', 'Coherence', 'Groundedness'],
            'Mean': [
                f"{golden_eval['summary']['metrics']['similarity']['mean']:.3f}",
                f"{golden_eval['summary']['metrics']['relevance']['mean']:.1f}/5",
                f"{golden_eval['summary']['metrics']['coherence']['mean']:.1f}/5",
                f"{golden_eval['summary']['metrics']['groundedness']['mean']:.1f}/5"
            ],
            'Range': [
                f"{golden_eval['summary']['metrics']['similarity']['min']:.3f} - {golden_eval['summary']['metrics']['similarity']['max']:.3f}",
                f"{golden_eval['summary']['metrics']['relevance']['min']}-{golden_eval['summary']['metrics']['relevance']['max']}/5",
                f"{golden_eval['summary']['metrics']['coherence']['min']}-{golden_eval['summary']['metrics']['coherence']['max']}/5",
                f"{golden_eval['summary']['metrics']['groundedness']['min']}-{golden_eval['summary']['metrics']['groundedness']['max']}/5"
            ]
        }
        
        df_metrics = pd.DataFrame(metrics_summary)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Radar chart for metrics
        st.markdown("### Metric Scores (Normalized to 0-1)")
        
        metrics_data = {
            'Metric': ['Similarity', 'Relevance', 'Coherence', 'Groundedness'],
            'Score': [
                golden_eval['summary']['metrics']['similarity']['mean'],
                golden_eval['summary']['metrics']['relevance']['mean'] / 5,
                golden_eval['summary']['metrics']['coherence']['mean'] / 5,
                golden_eval['summary']['metrics']['groundedness']['mean'] / 5,
            ]
        }
        
        fig = go.Figure(data=go.Scatterpolar(
            r=metrics_data['Score'],
            theta=metrics_data['Metric'],
            fill='toself',
            fillcolor='rgba(52, 152, 219, 0.5)',
            line=dict(color='rgba(52, 152, 219, 1)'),
            hovertemplate='<b>%{theta}</b><br>Score: %{r:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Generation Quality Metrics (Normalized)",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Overall quality
        st.markdown("### Overall Quality Assessment")
        
        overall_score = golden_eval['summary']['overall_score']
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if overall_score > 0.80:
                quality = "EXCELLENT"
                color = "green"
            elif overall_score > 0.70:
                quality = "GOOD"
                color = "blue"
            elif overall_score > 0.60:
                quality = "FAIR"
                color = "orange"
            else:
                quality = "NEEDS IMPROVEMENT"
                color = "red"
            
            st.markdown(f"**Overall Quality Score: {overall_score:.3f}/1.0**")
            st.markdown(f"**Assessment: {quality}**")
        
        with col2:
            st.metric("Potential Score", "0.85+/1.0", "With improvements")
    
    # =====================================================================
    # PAGE 4: COMBINED ANALYSIS
    # =====================================================================
    elif page == "📋 Combined Analysis":
        st.markdown("## System Performance Overview")
        
        # Create comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔍 Retrieval")
            st.markdown(f"""
            **Accuracy**: {auto_eval['summary']['accuracy']:.1%}
            
            - 77.9% of queries retrieved correct chunk
            - MRR: {auto_eval['summary']['mean_reciprocal_rank']:.2f}
            - Scale: 249 questions
            """)
        
        with col2:
            st.markdown("### 💎 Generation")
            st.markdown(f"""
            **Similarity**: {golden_eval['summary']['metrics']['similarity']['mean']:.3f}
            
            - Good semantic alignment
            - Relevance: 3.0/5
            - Coherence: 3.0/5
            """)
        
        with col3:
            st.markdown("### 📊 Overall")
            st.markdown(f"""
            **Quality**: {golden_eval['summary']['overall_score']:.3f}/1.0
            
            - Assessment: FAIR
            - Can improve to 0.85+
            - Strong retrieval foundation
            """)
        
        st.markdown("---")
        
        # System performance chart
        st.markdown("### System Performance Comparison")
        
        perf_data = {
            'Category': ['Retrieval\nAccuracy', 'Semantic\nSimilarity', 'Overall\nQuality'],
            'Score': [
                auto_eval['summary']['accuracy'],
                golden_eval['summary']['metrics']['similarity']['mean'],
                golden_eval['summary']['overall_score']
            ]
        }
        
        fig = px.bar(
            x=perf_data['Category'],
            y=perf_data['Score'],
            title='System Performance Metrics',
            labels={'y': 'Score (0-1)', 'x': ''},
            color=perf_data['Score'],
            color_continuous_scale=['#e74c3c', '#f39c12', '#2ecc71'],
            text=[f"{s:.2%}" for s in perf_data['Score']]
        )
        fig.update_traces(textposition='outside')
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("### 💡 Recommendations for Improvement")
        
        st.markdown("""
        **SHORT-TERM (Quick Wins):**
        1. Improve system prompt with clearer instructions
        2. Add few-shot examples for better guidance
        3. Enforce source citations
        
        **MEDIUM-TERM (Structural):**
        1. Implement hybrid search (semantic + keyword)
        2. Fine-tune on domain data
        3. Add answer validation layer
        
        **LONG-TERM (Strategic):**
        1. Production deployment with caching
        2. Domain specialization for Apple
        3. Multi-turn conversation support
        """)
    
    # =====================================================================
    # PAGE 5: FULL REPORT
    # =====================================================================
    elif page == "📑 Full Report":
        st.markdown("## Comprehensive Case Study Report")
        
        report_text = load_report()
        if report_text:
            st.text_area(
                "Full Report",
                value=report_text,
                height=600,
                disabled=True,
                label_visibility="collapsed"
            )
            
            # Download button
            st.download_button(
                label="📥 Download Report as TXT",
                data=report_text,
                file_name="COMPREHENSIVE_CASE_STUDY_REPORT.txt",
                mime="text/plain"
            )
        else:
            st.error("Could not load report")
    
    # =====================================================================
    # PAGE 6: ABOUT
    # =====================================================================
    elif page == "ℹ️ About":
        st.markdown("## About This Case Study")
        
        st.markdown("""
        ### 📊 Project Overview
        
        This is a comprehensive case study of a **Retrieval-Augmented Generation (RAG)** system
        built to analyze Apple's organizational structure and innovation practices.
        
        ### 🎯 Objectives
        
        1. Implement a production-ready RAG system
        2. Evaluate retrieval quality at scale (249 questions)
        3. Assess generation quality on expert-curated queries (15 queries)
        4. Provide comprehensive analysis and recommendations
        
        ### 📈 Evaluation Approach
        
        **Dual-Track Evaluation:**
        - **Manual Golden Dataset**: 15 expert-curated queries (4 metrics)
        - **Auto-Generated Questions**: 249 questions from chunks (scale testing)
        
        **Metrics:**
        - Retrieval: Accuracy@3, MRR
        - Generation: Similarity, Relevance, Coherence, Groundedness
        
        ### 🏗️ System Architecture
        
        1. **Document Processing**
           - Input: HBR Apple organizational PDF (11 pages)
           - Chunking: 500 tokens, 50-token overlap
           - Result: ~50 document chunks
        
        2. **Embedding & Retrieval**
           - Model: all-mpnet-base-v2 (sentence-transformers)
           - Search: Semantic similarity
           - Top-k: 3 results
        
        3. **Generation**
           - Model: Claude 3.5 Sonnet
           - Task: Answer generation grounded in context
        
        4. **Evaluation**
           - Auto-generated questions for scale testing
           - Manual golden dataset for quality assessment
           - 4 metrics for comprehensive evaluation
        
        ### 📁 Deliverables
        
        - ✅ RAG System Implementation
        - ✅ 249 Auto-Generated Questions
        - ✅ 4-Metric Evaluation Framework
        - ✅ Comprehensive Case Study Report
        - ✅ Interactive Dashboard with Chatbot
        
        ### 🔗 Links
        
        - **GitHub**: [rag-case-study](https://github.com/sushiva/rag-case-study)
        - **Report**: COMPREHENSIVE_CASE_STUDY_REPORT.txt
        - **Data**: results/ folder
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 12px;">
    RAG Case Study - Apple Organizational Model | 
    Retrieval Accuracy: 77.9% | 
    Overall Quality: 0.647/1.0 | 
    Interactive Chatbot Available 💬
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()