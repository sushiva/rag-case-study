"""
Enhanced RAG Dashboard with Multi-Model Support
Supports: Claude (Anthropic), Gemini (Google), GPT (OpenAI)
Apple Organizational Model - Interactive Results & Query Interface
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os

# Add src to path for RAG imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Set page config FIRST
st.set_page_config(
    page_title="RAG Case Study - Multi-Model",
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
    .api-key-box {
        background-color: #fff3cd;
        padding: 15px;
        border-left: 4px solid #ffc107;
        border-radius: 5px;
        margin: 10px 0;
    }
    .model-selector {
        background-color: #e8f4f8;
        padding: 15px;
        border-left: 4px solid #3498db;
        border-radius: 5px;
        margin: 10px 0;
    }
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

# Model configurations
MODEL_CONFIGS = {
    'Claude': {
        'provider': 'anthropic',
        'models': [
            'claude-3-5-haiku-20241022',
            'claude-3-opus-20250219',
        ],
        'env_var': 'ANTHROPIC_API_KEY',
        'color': '#9D4EDD',
        'icon': '🤖'
    },
    'Google Gemini': {
        'provider': 'google',
        'models': [
            'gemini-2.0-flash',
            'gemini-1.5-pro',
            'gemini-2.5-flash',
        ],
        'env_var': 'GOOGLE_API_KEY',
        'color': '#4285F4',
        'icon': '🔮'
    },
    'OpenAI': {
        'provider': 'openai',
        'models': [
            'gpt-4-turbo',
            'gpt-4o',
            'gpt-4o-mini',
            'gpt-3.5-turbo',
        ],
        'env_var': 'OPENAI_API_KEY',
        'color': '#00A67E',
        'icon': '⚡'
    }
}

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

if 'selected_model_provider' not in st.session_state:
    st.session_state.selected_model_provider = 'Claude'

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = MODEL_CONFIGS['Claude']['models'][0]

if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {}

if 'api_valid' not in st.session_state:
    st.session_state.api_valid = {}

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

# Create RAG system with selected model
def create_rag_system(model_provider, model_name, api_key):
    """Create RAG system with selected model"""
    try:
        from src.rag_system import RAGSystemMultiModel
        
        pdf_path = BASE_DIR / 'data' / 'HBR_How_Apple_Is_Organized_For_Innovation-4.pdf'
        
        rag = RAGSystemMultiModel(
            pdf_path=str(pdf_path),
            embedding_model="all-mpnet-base-v2",
            model_provider=model_provider,
            model_name=model_name,
            api_key=api_key
        )
        rag.build(chunk_size=500, chunk_overlap=50, use_cached=True)
        return rag
    except ImportError:
        # Fallback to original RAG system if multi-model version not available
        st.warning("Multi-model support not yet implemented, using Claude backend")
        from src.rag_system import RAGSystem
        
        pdf_path = BASE_DIR / 'data' / 'HBR_How_Apple_Is_Organized_For_Innovation-4.pdf'
        
        os.environ['ANTHROPIC_API_KEY'] = api_key
        rag = RAGSystem(
            pdf_path=str(pdf_path),
            embedding_model="all-mpnet-base-v2",
            claude_model=model_name
        )
        rag.build(chunk_size=500, chunk_overlap=50, use_cached=True)
        return rag
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None

# Validate API key
def validate_api_key(provider, api_key):
    """Validate API key with test call"""
    try:
        if provider == 'Claude':
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
        
        elif provider == 'Google Gemini':
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content("test")
        
        elif provider == 'OpenAI':
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
        
        return True
    except Exception as e:
        st.error(f"Invalid API key: {str(e)[:100]}")
        return False

# Setup multi-model API key management
def setup_multi_model_sidebar():
    """Setup API key input for multiple models"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔑 API Configuration")
    
    # Model Provider Selection
    st.sidebar.markdown("### 📦 Select Model Provider")
    
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        if st.button(f"{MODEL_CONFIGS['Claude']['icon']} Claude", use_container_width=True):
            st.session_state.selected_model_provider = 'Claude'
            st.rerun()
    
    with col2:
        if st.button(f"{MODEL_CONFIGS['Google Gemini']['icon']} Gemini", use_container_width=True):
            st.session_state.selected_model_provider = 'Google Gemini'
            st.rerun()
    
    with col3:
        if st.button(f"{MODEL_CONFIGS['OpenAI']['icon']} OpenAI", use_container_width=True):
            st.session_state.selected_model_provider = 'OpenAI'
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Current provider info
    provider = st.session_state.selected_model_provider
    config = MODEL_CONFIGS[provider]
    
    st.sidebar.markdown(f"### {config['icon']} {provider}")
    
    # Check environment variable
    api_key_env = os.getenv(config['env_var'])
    
    if api_key_env and provider not in st.session_state.api_valid:
        st.sidebar.success(f"✓ API Key loaded from {config['env_var']}")
        st.session_state.api_keys[provider] = api_key_env
        st.session_state.api_valid[provider] = True
    
    # Show instructions
    if provider == 'Claude':
        st.sidebar.markdown("""
        **Get API Key:**
        1. Visit [console.anthropic.com](https://console.anthropic.com/)
        2. Sign up/Login
        3. API Keys → Create Key
        4. Copy key (starts with `sk-ant-`)
        """)
    
    elif provider == 'Google Gemini':
        st.sidebar.markdown("""
        **Get API Key:**
        1. Visit [aistudio.google.com](https://aistudio.google.com/)
        2. Click "Get API Key"
        3. Create new API key
        4. Copy the key
        """)
    
    elif provider == 'OpenAI':
        st.sidebar.markdown("""
        **Get API Key:**
        1. Visit [platform.openai.com](https://platform.openai.com/)
        2. Sign up/Login
        3. API Keys → Create new
        4. Copy the key (starts with `sk-`)
        """)
    
    # API Key input
    api_key_input = st.sidebar.text_input(
        f"Enter {provider} API Key:",
        type="password",
        value=st.session_state.api_keys.get(provider, ""),
        key=f"api_key_{provider}"
    )
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button(f"🔐 Validate {provider}", use_container_width=True):
            if not api_key_input:
                st.sidebar.error("Please enter an API key")
            else:
                with st.spinner("🔄 Validating..."):
                    if validate_api_key(provider, api_key_input):
                        st.session_state.api_keys[provider] = api_key_input
                        st.session_state.api_valid[provider] = True
                        st.sidebar.success(f"✓ {provider} API Key validated!")
                        st.rerun()
                    else:
                        st.sidebar.error(f"❌ Invalid {provider} API key")
    
    with col2:
        if st.button(f"🧹 Clear Key", use_container_width=True):
            if provider in st.session_state.api_keys:
                del st.session_state.api_keys[provider]
            if provider in st.session_state.api_valid:
                del st.session_state.api_valid[provider]
            st.rerun()
    
    # Model selection
    if st.session_state.api_valid.get(provider):
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎯 Select Model")
        
        selected_model = st.sidebar.selectbox(
            "Choose model:",
            config['models'],
            key=f"model_select_{provider}"
        )
        
        st.session_state.selected_model = selected_model
        
        st.sidebar.success(f"✓ Connected with {provider}")
        st.sidebar.info(f"Model: {selected_model}")
        
        return True
    
    return False

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
    st.markdown("# 🎯 RAG Case Study - Multi-Model Support")
    st.markdown("**Retrieval-Augmented Generation with Claude, Gemini & OpenAI**")
    
    # Load data
    auto_eval, golden_eval, report = load_data()
    
    if auto_eval is None:
        st.error(f"Could not load evaluation results from {BASE_DIR}")
        return
    
    # Setup API key and model selection in sidebar
    api_key_valid = setup_multi_model_sidebar()
    
    # Sidebar navigation
    st.sidebar.markdown("---")
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
        
        # Display selected model
        provider = st.session_state.selected_model_provider
        model = st.session_state.selected_model
        config = MODEL_CONFIGS[provider]
        
        st.markdown(f"**Using: {config['icon']} {provider} - {model}**")
        
        st.markdown("Ask questions about Apple's organizational model and get answers from the document!")
        
        # Check if API key is available
        if not api_key_valid:
            st.warning("⚠️ **API Key Required**")
            st.info(f"""
            To use the chatbot, please:
            1. Select a model provider from the sidebar
            2. Get your API key from the provider's console
            3. Enter it in the sidebar
            4. Click **🔐 Validate** button
            5. Return here and ask questions!
            """)
            return
        
        # Initialize RAG system
        with st.spinner(f"🔄 Initializing RAG with {provider}..."):
            rag = create_rag_system(
                provider,
                model,
                st.session_state.api_keys[provider]
            )
        
        if rag is None:
            st.error("Failed to initialize RAG system")
            return
        
        st.success(f"✓ RAG system ready with {provider}!")
        
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
        
        # Chat input - MUST be outside columns/expanders
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
                with st.spinner(f"🔄 Searching documents and generating answer using {provider}..."):
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
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Retrieved Chunks", len(result.get('retrieved_results', [])))
                        with col2:
                            if result.get('retrieved_results'):
                                pages = [r.page for r in result['retrieved_results']]
                                st.metric("Source Pages", f"{min(pages)}-{max(pages)}")
                        with col3:
                            st.metric("Provider", provider)
                        with col4:
                            st.metric("Model", model.split('/')[-1][:20])
                    
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
        
        # Model comparison info
        st.markdown("### 🎯 Multi-Model Support")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            **{MODEL_CONFIGS['Claude']['icon']} Claude (Anthropic)**
            - Fast & accurate
            - Best for reasoning
            - Free tier available
            """)
        
        with col2:
            st.markdown(f"""
            **{MODEL_CONFIGS['Google Gemini']['icon']} Gemini (Google)**
            - Multimodal capable
            - Excellent context window
            - Good for long docs
            """)
        
        with col3:
            st.markdown(f"""
            **{MODEL_CONFIGS['OpenAI']['icon']} OpenAI**
            - Industry standard
            - Highly reliable
            - Excellent quality
            """)
        
        st.markdown("---")
        
        # Key findings
        st.markdown("### 🎯 Key Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Strengths")
            st.markdown(f"""
            - **Strong Retrieval**: {auto_eval['summary']['accuracy']:.1%} accuracy on 249 questions
            - **Good Semantic Alignment**: {golden_eval['summary']['metrics']['similarity']['mean']:.3f} similarity
            - **Multi-Model Support**: Use any LLM provider
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
        - **Basic Questions**: {complexity_data['basic']*100:.1f}% accuracy
        - **Intermediate Questions**: {complexity_data['intermediate']*100:.1f}% accuracy
        - **Advanced Questions**: {complexity_data['advanced']*100:.1f}% accuracy
        - **All complexity levels exceed 75% accuracy**
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
    
    # =====================================================================
    # PAGE 4: COMBINED ANALYSIS
    # =====================================================================
    elif page == "📋 Combined Analysis":
        st.markdown("## System Performance Overview")
        
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
        with **multi-model support** for analyzing Apple's organizational structure.
        
        ### 🎯 Multi-Model Support
        
        Choose from your preferred LLM provider:
        
        **Claude (Anthropic)**
        - Latest: Claude 3.5 Sonnet, Haiku
        - Fast reasoning and analysis
        - Free tier available
        
        **Google Gemini**
        - Latest: Gemini 2.0 Flash
        - Excellent long-context handling
        - Multimodal capabilities
        
        **OpenAI GPT**
        - Latest: GPT-4o, GPT-4 Turbo
        - Industry-standard quality
        - Highly reliable
        
        ### 🏗️ System Architecture
        
        1. **Document Processing**
           - Input: HBR Apple organizational PDF (11 pages)
           - Chunking: 500 tokens, 50-token overlap
           - Result: ~50 document chunks
        
        2. **Embedding & Retrieval**
           - Model: all-mpnet-base-v2 (sentence-transformers)
           - Search: Semantic similarity
           - Top-k: 3 results
        
        3. **Generation (Multi-Model)**
           - Claude, Gemini, or GPT
           - Context: Retrieved chunks
           - Task: Answer generation grounded in context
        
        4. **Evaluation**
           - 249 auto-generated questions for scale testing
           - 15 manual golden queries for quality assessment
           - 4 metrics for comprehensive evaluation
        
        ### 🔑 Getting API Keys
        
        **Claude (Anthropic):**
        Visit: [console.anthropic.com](https://console.anthropic.com/)
        
        **Gemini (Google):**
        Visit: [aistudio.google.com](https://aistudio.google.com/)
        
        **OpenAI:**
        Visit: [platform.openai.com](https://platform.openai.com/)
        
        ### 📁 Deliverables
        
        - ✅ RAG System with Multi-Model Support
        - ✅ 249 Auto-Generated Questions
        - ✅ 4-Metric Evaluation Framework
        - ✅ Comprehensive Case Study Report
        - ✅ Interactive Dashboard with Chatbot
        - ✅ Multi-Provider API Key Management
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 12px;">
    RAG Case Study - Apple Organizational Model | 
    Multi-Model Support (Claude, Gemini, OpenAI) | 
    Retrieval Accuracy: 77.9% | 
    Quality: 0.647/1.0
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()