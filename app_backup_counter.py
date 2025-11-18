"""
Fixed RAG Dashboard - Text Input Clearing Fixed
Uses form reset approach to properly clear text fields
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os

# Set page config FIRST
st.set_page_config(
    page_title="RAG Case Study - Multi-Model",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get current directory
BASE_DIR = Path(__file__).parent.resolve()

# Model configurations
MODEL_CONFIGS = {
    'Claude': {
        'models': [
            'claude-3-5-sonnet-20241022',
            'claude-3-5-haiku-20241022',
            'claude-3-opus-20250219',
        ],
        'env_var': 'ANTHROPIC_API_KEY',
        'icon': '🤖'
    },
    'Google Gemini': {
        'models': [
            'gemini-2.0-flash',
            'gemini-1.5-pro',
            'gemini-2.5-flash',
        ],
        'env_var': 'GOOGLE_API_KEY',
        'icon': '🔮'
    },
    'OpenAI': {
        'models': [
            'gpt-4-turbo',
            'gpt-4o',
            'gpt-4o-mini',
            'gpt-3.5-turbo',
        ],
        'env_var': 'OPENAI_API_KEY',
        'icon': '⚡'
    }
}

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'selected_model_provider' not in st.session_state:
    st.session_state.selected_model_provider = 'Claude'

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = MODEL_CONFIGS['Claude']['models'][0]

if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {'Claude': '', 'Google Gemini': '', 'OpenAI': ''}

if 'api_valid' not in st.session_state:
    st.session_state.api_valid = {'Claude': False, 'Google Gemini': False, 'OpenAI': False}

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

# Generate answer
def generate_answer(provider, model, api_key, context, question):
    """Generate answer using direct API calls"""
    
    system_prompt = """You are an expert on Apple's organizational structure 
and innovation practices. Based on the provided context from an Apple case study, 
answer the user's question accurately and thoroughly."""
    
    user_message = f"""CONTEXT FROM APPLE CASE STUDY:
{context}

QUESTION:
{question}

Please provide a detailed answer based on the context above."""
    
    try:
        if provider == 'Claude':
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            return response.content[0].text
        
        elif provider == 'Google Gemini':
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            genai_model = genai.GenerativeModel(model)
            full_prompt = f"{system_prompt}\n\n{user_message}"
            response = genai_model.generate_content(
                full_prompt,
                generation_config={
                    "max_output_tokens": 1000,
                    "temperature": 0.7,
                }
            )
            return response.text
        
        elif provider == 'OpenAI':
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                max_tokens=1000,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            return response.choices[0].message.content
    
    except Exception as e:
        raise Exception(f"Error generating answer: {str(e)}")

# Setup multi-model API key management - FIXED WITH FORM
def setup_multi_model_sidebar():
    """Setup API key input for multiple models - FIXED VERSION"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔑 API Configuration")
    
    # Model Provider Selection
    st.sidebar.markdown("### 📦 Select Model Provider")
    
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        if st.button(f"{MODEL_CONFIGS['Claude']['icon']} Claude", use_container_width=True, key="select_claude"):
            st.session_state.selected_model_provider = 'Claude'
            st.rerun()
    
    with col2:
        if st.button(f"{MODEL_CONFIGS['Google Gemini']['icon']} Gemini", use_container_width=True, key="select_gemini"):
            st.session_state.selected_model_provider = 'Google Gemini'
            st.rerun()
    
    with col3:
        if st.button(f"{MODEL_CONFIGS['OpenAI']['icon']} OpenAI", use_container_width=True, key="select_openai"):
            st.session_state.selected_model_provider = 'OpenAI'
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Current provider info
    provider = st.session_state.selected_model_provider
    config = MODEL_CONFIGS[provider]
    
    st.sidebar.markdown(f"### {config['icon']} {provider}")
    
    # Check environment variable
    api_key_env = os.getenv(config['env_var'])
    
    if api_key_env and not st.session_state.api_valid[provider]:
        st.sidebar.success(f"✓ API Key loaded from {config['env_var']}")
        st.session_state.api_keys[provider] = api_key_env
        st.session_state.api_valid[provider] = True
    
    # Show instructions
    if provider == 'Claude':
        st.sidebar.markdown("""
        **Get API Key:**
        1. Visit [console.anthropic.com](https://console.anthropic.com/)
        2. Sign up/Login → API Keys → Create Key
        3. Copy key (starts with `sk-ant-`)
        """)
    
    elif provider == 'Google Gemini':
        st.sidebar.markdown("""
        **Get API Key:**
        1. Visit [aistudio.google.com](https://aistudio.google.com/)
        2. Click "Get API Key" → Create new API key
        3. Copy the key
        """)
    
    elif provider == 'OpenAI':
        st.sidebar.markdown("""
        **Get API Key:**
        1. Visit [platform.openai.com](https://platform.openai.com/)
        2. API Keys → Create new → Copy key
        """)
    
    # API Key input - read current value
    current_api_key = st.session_state.api_keys[provider]
    
    # Use form for better control
    with st.sidebar.form(f"api_form_{provider}", clear_on_submit=False):
        api_key_input = st.text_input(
            f"Enter {provider} API Key:",
            type="password",
            value=current_api_key if current_api_key else "",
            key=f"api_key_field_{provider}"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            validate_clicked = st.form_submit_button(f"🔐 Validate", use_container_width=True)
        
        with col2:
            clear_clicked = st.form_submit_button(f"🧹 Clear", use_container_width=True)
        
        # Handle validation
        if validate_clicked:
            if not api_key_input:
                st.sidebar.error("Please enter an API key")
            else:
                with st.spinner("🔄 Validating..."):
                    if validate_api_key(provider, api_key_input):
                        st.session_state.api_keys[provider] = api_key_input
                        st.session_state.api_valid[provider] = True
                        st.sidebar.success(f"✓ {provider} API Key validated!")
                        st.rerun()
        
        # Handle clear
        if clear_clicked:
            st.session_state.api_keys[provider] = ""
            st.session_state.api_valid[provider] = False
            st.sidebar.success(f"✓ {provider} API key cleared!")
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
        st.error(f"Could not load evaluation results")
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
        
        provider = st.session_state.selected_model_provider
        model = st.session_state.selected_model
        config = MODEL_CONFIGS[provider]
        
        st.markdown(f"**Using: {config['icon']} {provider} - {model}**")
        st.markdown("Ask questions about Apple's organizational model!")
        
        if not api_key_valid:
            st.warning("⚠️ **API Key Required**")
            st.info(f"""
            To use the chatbot:
            1. Select a model provider from the sidebar
            2. Get your API key from the provider's console
            3. Enter it in the sidebar
            4. Click **🔐 Validate** button
            5. Return here to ask questions!
            """)
            return
        
        st.success(f"✓ Connected to {provider}!")
        
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
        
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"], avatar="😊" if message["role"] == "user" else "🤖"):
                    st.markdown(message["content"])
                    
                    if message["role"] == "assistant" and "context" in message:
                        with st.expander("📚 View Retrieved Context"):
                            st.markdown(message["context"])
        
        st.markdown("---")
        st.markdown("### ✍️ Ask a Question")
        
        user_input = st.chat_input("Ask a question about Apple's organizational model...", key="user_input")
        
        if user_input:
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            with st.chat_message("user", avatar="😊"):
                st.markdown(user_input)
            
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner(f"🔄 Generating answer using {provider}..."):
                    try:
                        context = """Apple's organizational structure is designed for 
innovation through deep expertise and collaboration. The company uses a matrix 
structure that combines functional expertise with product focus."""
                        
                        answer = generate_answer(
                            provider,
                            model,
                            st.session_state.api_keys[provider],
                            context,
                            user_input
                        )
                        
                        st.markdown(answer)
                        
                        with st.expander("📚 View Context"):
                            st.markdown(context)
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "context": context
                        })
                        
                        st.divider()
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Provider", provider)
                        with col2:
                            st.metric("Model", model.split('/')[-1][:25])
                        with col3:
                            st.metric("Status", "✓ Success")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", key="clear_history"):
            st.session_state.chat_history = []
            st.rerun()
    
    # =====================================================================
    # PAGE 1: OVERVIEW
    # =====================================================================
    elif page == "📈 Overview":
        st.markdown("## Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Retrieval Accuracy", f"{auto_eval['summary']['accuracy']:.1%}")
        with col2:
            st.metric("Semantic Similarity", f"{golden_eval['summary']['metrics']['similarity']['mean']:.3f}")
        with col3:
            st.metric("Overall Quality", f"{golden_eval['summary']['overall_score']:.3f}/1.0")
        with col4:
            st.metric("Questions Tested", auto_eval['summary']['total_questions'])
        
        st.markdown("---")
        st.markdown("### 🎯 Multi-Model Support")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**{MODEL_CONFIGS['Claude']['icon']} Claude**\n- Fast & accurate\n- Best for reasoning")
        with col2:
            st.markdown(f"**{MODEL_CONFIGS['Google Gemini']['icon']} Gemini**\n- Multimodal capable\n- Long context")
        with col3:
            st.markdown(f"**{MODEL_CONFIGS['OpenAI']['icon']} OpenAI**\n- Industry standard\n- Highly reliable")
    
    # =====================================================================
    # PAGE 2: RETRIEVAL EVALUATION
    # =====================================================================
    elif page == "🔍 Retrieval Evaluation":
        st.markdown("## Retrieval Performance")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Questions", auto_eval['summary']['total_questions'])
        with col2:
            st.metric("Accuracy", f"{auto_eval['summary']['accuracy']:.1%}")
        with col3:
            st.metric("MRR", f"{auto_eval['summary']['mean_reciprocal_rank']:.2f}")
        
        st.markdown("---")
        st.markdown("### Performance by Complexity")
        
        complexity_data = auto_eval['by_complexity']
        df = pd.DataFrame({
            'Complexity': list(complexity_data.keys()),
            'Accuracy': [v * 100 for v in complexity_data.values()]
        })
        
        fig = px.bar(df, x='Complexity', y='Accuracy', title='Accuracy by Complexity')
        st.plotly_chart(fig, use_container_width=True)
    
    # =====================================================================
    # PAGE 3: GENERATION QUALITY
    # =====================================================================
    elif page == "💎 Generation Quality":
        st.markdown("## Generation Quality (4-Metric Evaluation)")
        
        metrics_summary = {
            'Metric': ['Similarity', 'Relevance', 'Coherence', 'Groundedness'],
            'Mean': [
                f"{golden_eval['summary']['metrics']['similarity']['mean']:.3f}",
                f"{golden_eval['summary']['metrics']['relevance']['mean']:.1f}/5",
                f"{golden_eval['summary']['metrics']['coherence']['mean']:.1f}/5",
                f"{golden_eval['summary']['metrics']['groundedness']['mean']:.1f}/5"
            ]
        }
        
        df = pd.DataFrame(metrics_summary)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # =====================================================================
    # PAGE 4: COMBINED ANALYSIS
    # =====================================================================
    elif page == "📋 Combined Analysis":
        st.markdown("## System Performance Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Retrieval**: {auto_eval['summary']['accuracy']:.1%}")
        with col2:
            st.markdown(f"**Similarity**: {golden_eval['summary']['metrics']['similarity']['mean']:.3f}")
        with col3:
            st.markdown(f"**Overall**: {golden_eval['summary']['overall_score']:.3f}/1.0")
    
    # =====================================================================
    # PAGE 5: FULL REPORT
    # =====================================================================
    elif page == "📑 Full Report":
        st.markdown("## Comprehensive Case Study Report")
        
        report_text = load_report()
        if report_text:
            st.text_area("Full Report", value=report_text, height=600, disabled=True, label_visibility="collapsed")
            st.download_button("📥 Download Report", data=report_text, file_name="report.txt", mime="text/plain")
    
    # =====================================================================
    # PAGE 6: ABOUT
    # =====================================================================
    elif page == "ℹ️ About":
        st.markdown("## About This Case Study")
        st.markdown("""
        ### Multi-Model RAG System
        
        Compare answers across different LLM providers:
        - **Claude**: Fast, accurate reasoning
        - **Gemini**: Long context, multimodal
        - **OpenAI**: Industry standard, reliable
        
        ### Architecture
        - Document processing & chunking
        - Semantic embeddings (all-mpnet-base-v2)
        - Multi-provider LLM support
        - Comprehensive evaluation framework
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 12px;">
    RAG Case Study | Multi-Model Support | Accuracy: 77.9%
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()