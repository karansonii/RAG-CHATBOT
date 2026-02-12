#main.py
import sys
import os
import glob
import streamlit as st
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
import streamlit.components.v1 as components
import threading
import json
import time
import re
# Load environment first
load_dotenv()
# Setup project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Import core modules
from src.utils import ConfigManager, validate_api_keys, fuzzy_match_text
from src.embedder import create_or_load_index
from src.rag_pipeline import rag_answer
from src.voice_modules import play_tts, load_domain_keywords, LiveMicRecorder, DOMAIN_KEYWORDS
from src.translator import detect_language, translate_text, ALLOWED_LANGUAGES
from src.visual_generator import process_visual_request
from src.data_loader import load_documents_from_folder
from src.system_prompt import SystemPrompts
from src.chat_history import init_chat_history
# Initialize system prompts
prompts = SystemPrompts()
# Validate API keys on startup
try:
    validate_api_keys()
except EnvironmentError as e:
    st.error(f"⚠️ Configuration Error: {e}")
    st.stop()
# Load configuration using ConfigManager
model_config = ConfigManager.get_model_config()
API_KEY = ConfigManager.get_api_key('gemini')
MODEL_NAME = model_config['primary_model']
FALLBACK_MODEL = model_config['fallback_model']
# Get paths
DATA_FOLDER = str(ConfigManager.get_path('DATA_FOLDER', './data/'))
VECTOR_DB_PATH = str(ConfigManager.get_path('VECTOR_DB_PATH', './data/vectorstore'))
# Auto-load all files from data folder
DOCS_PATHS = []
supported_extensions = ConfigManager.get('SUPPORTED_FILE_EXTENSIONS', '*.docx,*.txt,*.pdf,*.pptx,*.ppt,*.csv,*.md')
for ext in supported_extensions.split(','):
    DOCS_PATHS += glob.glob(os.path.join(DATA_FOLDER, ext))
DOCS_PATHS = [p.strip() for p in DOCS_PATHS]
# Configure Gemini
genai.configure(api_key=API_KEY)
# ------------------------
# Initialize Chat History Manager
# ------------------------
chat_history_manager = init_chat_history()
# Run cleanup on startup if enabled
if os.getenv("CHAT_HISTORY_AUTO_CLEANUP", "true").lower() == "true":
    cleaned = chat_history_manager.cleanup_old_history()
    if cleaned > 0:
        print(f"🗑️ Cleaned up {cleaned} old chat history files")
# ------------------------
# Streamlit Page Config
# ------------------------
st.set_page_config(
    page_title=ConfigManager.get('APP_TITLE', 'AI Chatbot Assistant'),
    layout=ConfigManager.get('APP_LAYOUT', 'wide'),
    initial_sidebar_state=ConfigManager.get('SIDEBAR_STATE', 'expanded'),
    menu_items=None
)
# ------------------------
# MODERN CSS DESIGN - FIXED INPUT BAR
# ------------------------
st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
   
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
   
    /* Remove default Streamlit padding/margins and fix scrolling */
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
        background: #ffffff;
        position: relative !important;
        min-height: 100vh !important;
    }
   
    /* Prevent main content from scrolling over fixed elements */
    .main {
        overflow-x: hidden !important;
    }
   
    .stApp {
        overflow-x: hidden !important;
        background: #ffffff !important;
    }
   
    /* Global Typography */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
   
    /* ============ SIDEBAR REDESIGN ============ */
    section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        width: 300px !important;
        min-width: 300px !important;
        background: linear-gradient(180deg, #1a1f36 0%, #0f1419 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
   
    section[data-testid="stSidebar"] > div {
        padding: 24px 20px;
    }
   
    /* Sidebar Title */
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        margin-bottom: 24px !important;
        text-transform: uppercase;
    }
   
    /* Sidebar Sections */
    .sidebar-section {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 16px;
        overflow: hidden;
        transition: all 0.3s ease;
    }
   
    .sidebar-section:hover {
        border-color: rgba(79, 209, 197, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 209, 197, 0.15);
    }
   
    /* Sidebar Buttons */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: rgba(79, 209, 197, 0.1) !important;
        color: #4FD1C5 !important;
        border: 1px solid rgba(79, 209, 197, 0.3) !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        height: auto !important;
        min-height: 48px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
        text-align: left !important;
    }
   
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(79, 209, 197, 0.2) !important;
        border-color: #4FD1C5 !important;
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(79, 209, 197, 0.2);
    }
   
    /* Sidebar Text */
    section[data-testid="stSidebar"] .element-container {
        color: rgba(255,255,255,0.9) !important;
    }
   
    section[data-testid="stSidebar"] p {
        color: rgba(255,255,255,0.85) !important;
        font-size: 13px !important;
        line-height: 1.6 !important;
    }
   
    section[data-testid="stSidebar"] strong {
        color: #4FD1C5 !important;
    }
   
    /* ============ MAIN CONTENT AREA ============ */
   
    /* Header Section - FIXED */
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 32px 20px;
        text-align: center;
        border-bottom: 3px solid #5a67d8;
        position: fixed;
        top: 0;
        left: 300px;
        right: 0;
        z-index: 100;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
   
    .chat-header h1 {
        font-size: 36px !important;
        color: #ffffff !important;
        margin: 0 !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
   
    .chat-header p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 16px !important;
        margin-top: 8px !important;
        font-weight: 400;
    }
   
    /* Chat Container - SCROLLABLE AREA */
    .chat-wrapper {
        max-width: 850px; /* Match input bar width */
        margin: 0 auto;
        padding: 25px 20px 180px;
        min-height: auto;
        overflow-y: visible;
        position: relative;
    }
    .chunk-preview-wrapper {
    min-height: auto !important;
    padding: 40px 20px !important;
    }
   
    /* Empty State */
    .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    min-height: auto;
    max-width: 800px; /* NEW: match message width */
    margin: 0 auto; /* NEW: center it */
    text-align: center;
    animation: fadeIn 0.6s ease;
    padding-top: 0;
    }
   
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
   
    .empty-state-icon {
        font-size: 80px;
        margin-bottom: 24px;
        animation: float 3s ease-in-out infinite;
    }
   
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
   
    .empty-state h2 {
        font-size: 32px;
        color: #1a202c;
        margin-bottom: 16px;
        font-weight: 700;
    }
   
    .empty-state-subtitle {
        font-size: 16px;
        color: #718096;
        margin-bottom: 48px;
        max-width: 500px;
    }
   
    /* Example Prompts - Updated for Streamlit Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #f6f8fb 0%, #ffffff 100%) !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 10px 16px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        text-align: left !important;
        height: auto !important;
        min-height: 40px !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
        color: #2d3748 !important;
        font-weight: 500 !important;
    }
   
    .stButton > button:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.15) !important;
        border-color: #667eea !important;
        background: linear-gradient(135deg, #f0f2ff 0%, #ffffff 100%) !important;
    }
   
    /* ============ CHAT MESSAGES ============ */
    .message-row {
    display: flex;
    margin: 24px auto; /* Changed: auto centers horizontally */
    max-width: 800px; /* NEW: constrains message width */
    animation: slideIn 0.4s ease;
    }
   
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
   
    .message-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        flex-shrink: 0;
        margin-right: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
   
    .user-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
   
    .assistant-avatar {
        background: linear-gradient(135deg, #4FD1C5 0%, #3182CE 100%);
    }
   
    .message-content {
        flex: 1;
        padding: 18px 22px;
        border-radius: 18px;
        font-size: 15px;
        line-height: 1.7;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
   
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        border-bottom-right-radius: 4px;
    }
   
    .assistant-message {
        background: #f7fafc;
        color: #2d3748;
        border: 1px solid #e2e8f0;
        border-bottom-left-radius: 4px;
    }
   
    /* Message Formatting */
    .message-content p {
        margin: 10px 0 !important;
        line-height: 1.8 !important;
    }
   
    .message-content ul {
        margin: 16px 0 !important;
        padding-left: 28px !important;
        list-style-type: none !important;
    }
   
    .message-content ul li {
        margin-bottom: 14px !important;
        line-height: 1.7 !important;
        position: relative;
        padding-left: 8px;
    }
   
    .message-content ul li::before {
        content: "→";
        position: absolute;
        left: -24px;
        color: #4FD1C5;
        font-weight: bold;
        font-size: 16px;
    }
   
    .message-content strong {
        color: #667eea !important;
        font-weight: 600 !important;
    }
   
    /* TTS Button */
    .tts-button {
        margin-top: 12px !important;
    }
   
    .tts-button button {
        background: linear-gradient(135deg, #4FD1C5 0%, #3182CE 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 8px 18px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(79, 209, 197, 0.3) !important;
    }
   
    .tts-button button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(79, 209, 197, 0.4) !important;
    }
   
    /* ============ FIXED INPUT SECTION - CRITICAL FIX ============ */
   
    /* Hide the default columns container that Streamlit creates */
    div[data-testid="stHorizontalBlock"]:has(.input-text-wrapper) {
        position: fixed !important;
        bottom: 20px !important;
        left: 320px !important;
        right: 20px !important;
        z-index: 1000 !important;
        background: white !important;
        padding: 2px 25px !important;
        border-radius: 20px !important;
        box-shadow: 0 -2px 20px #667eea !important;
        max-width: 850px !important;
        margin: 0 auto !important;
        display: flex !important;
        align-items: center !important;
        gap: 15px !important;
    }
   
    /* Style for input column */
    .input-text-wrapper {
        flex: 1 !important;
        min-width: 0 !important;
    }
   
    /* Text input styling */
    .input-text-wrapper .stTextInput {
        margin: 0 !important;
    }
   
    .input-text-wrapper .stTextInput > div {
        margin: 0 !important;
    }
   
    .input-text-wrapper .stTextInput > div > div {
        margin: 0 !important;
    }
   
    .input-text-wrapper .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 14px 20px !important;
        font-size: 15px !important;
        box-shadow: none !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        box-sizing: border-box !important;
        height: 50px !important;
        background: white !important;
    }
   
    .input-text-wrapper .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }
   
    /* Mic button wrapper */
    .mic-btn-wrapper {
    flex-shrink: 0 !important;
    margin-left: -8px !important;
    margin-right: -8px !important;
    }
   
    .mic-btn-wrapper .stButton {
        margin: 0 !important;
    }
   
    .mic-btn-wrapper .stButton > button {
        min-height: 50px !important;
        width: 50px !important;
        height: 50px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        font-size: 20px !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(79, 209, 197, 0.3) !important;
        transition: all 0.3s ease !important;
        background: linear-gradient(135deg, #4FD1C5 0%, #3182CE 100%) !important;
        color: white !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
   
    .mic-btn-wrapper .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(79, 209, 197, 0.4) !important;
    }
   
    /* Send button wrapper */
    .send-btn-wrapper {
    flex-shrink: 0 !important;
    margin-left: -8px !important;
    }
   
    .send-btn-wrapper .stButton {
        margin: 0 !important;
    }
   
    .send-btn-wrapper .stButton > button {
        min-height: 50px !important;
        width: 50px !important;
        height: 50px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        font-size: 20px !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
   
    .send-btn-wrapper .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
        /* Aggressively remove all spacing between button columns */
        div[data-testid="stHorizontalBlock"]:has(.input-text-wrapper) > div[data-testid="column"]:nth-child(2) {
        margin-left: -12px !important;
        }
    div[data-testid="stHorizontalBlock"]:has(.input-text-wrapper) > div[data-testid="column"]:nth-child(3) {
        margin-left: -12px !important;
    }
    /* Force reduce gap between ALL columns in input bar */
        div[data-testid="stHorizontalBlock"]:has(.input-text-wrapper) > div[data-testid="column"] {
        padding: 0 !important;
        margin: 0 !important;
        gap: 0 !important;
    }
    /* Specifically target mic and send button columns */
        .mic-btn-wrapper,
        .send-btn-wrapper {
        padding: 0 !important;
        margin: 0 !important;
    }
    /* Override Streamlit's column gap */
        div[data-testid="stHorizontalBlock"]:has(.input-text-wrapper) {
        gap: 6px !important;
        column-gap: 24px !important;
    }
   
    /* Loading Animation */
        .loading-dots {
        display: inline-flex;
        gap: 6px;
        padding: 16px 20px;
    }
   
    .loading-dots span {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: linear-gradient(135deg, #4FD1C5 0%, #3182CE 100%);
        animation: bounce 1.4s infinite ease-in-out both;
    }
   
    .loading-dots span:nth-child(1) { animation-delay: -0.32s; }
    .loading-dots span:nth-child(2) { animation-delay: -0.16s; }
   
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); opacity: 0.5; }
        40% { transform: scale(1); opacity: 1; }
    }
   
    /* Recording State */
    .recording-indicator {
        background: linear-gradient(135deg, #fc8181 0%, #f56565 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 16px;
        text-align: center;
        margin: 20px auto;
        max-width: 600px;
        box-shadow: 0 4px 12px rgba(245, 101, 101, 0.3);
        animation: pulse 1.5s ease-in-out infinite;
    }
   
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
   
    .recording-indicator h3 {
        margin: 0 0 12px 0;
        font-size: 20px;
        font-weight: 600;
    }
   
    /* Responsive Design */
    @media (max-width: 1024px) {
        section[data-testid="stSidebar"] {
            width: 260px !important;
            min-width: 260px !important;
        }
       
        .chat-header {
            left: 260px !important;
        }
       
        div[data-testid="stHorizontalBlock"]:has(.input-text-wrapper) {
            left: 280px !important;
        }
       
        .chat-header h1 {
            font-size: 28px !important;
        }
    }
    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"]:has(.input-text-wrapper) {
            left: 12px !important;
            right: 12px !important;
            padding: 12px 16px !important;
        }
       
        .chat-header {
            left: 0 !important;
            padding: 24px 16px;
        }
       
        section[data-testid="stSidebar"] {
            width: 240px !important;
            min-width: 240px !important;
        }
       
        .message-avatar {
            width: 36px;
            height: 36px;
            font-size: 18px;
        }
       
        .chat-header h1 {
            font-size: 24px !important;
        }
       
        .mic-btn-wrapper .stButton > button,
        .send-btn-wrapper .stButton > button {
            min-height: 48px !important;
            width: 48px !important;
            font-size: 18px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Session State Initialization
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "🤖 AI Chatbot"
if "processing" not in st.session_state:
    st.session_state.processing = False
if "user_input_text" not in st.session_state:
    st.session_state.user_input_text = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "listening" not in st.session_state:
    st.session_state.listening = False
if "mic_recorder" not in st.session_state:
    st.session_state.mic_recorder = LiveMicRecorder()
if "mic_text" not in st.session_state:
    st.session_state.mic_text = ""
if "awaiting_followup" not in st.session_state:
    st.session_state.awaiting_followup = None
if "input_key" not in st.session_state:
    st.session_state.input_key = 0
if "show_full_visualization" not in st.session_state:
    st.session_state.show_full_visualization = False
if "current_visualization_data" not in st.session_state:
    st.session_state.current_visualization_data = None
if "awaiting_visualization" not in st.session_state:
    st.session_state.awaiting_visualization = False
if "awaiting_followup_type" not in st.session_state:
    st.session_state.awaiting_followup_type = None
if "last_rag_response" not in st.session_state:
    st.session_state.last_rag_response = ""
if "last_query_for_visuals" not in st.session_state:
    st.session_state.last_query_for_visuals = ""
if "last_input_language" not in st.session_state:
    st.session_state.last_input_language = "en"
if "last_input_language" not in st.session_state:
    st.session_state.last_input_language = "en"
# User authentication state - NEW
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "chat_history_loaded" not in st.session_state:
    st.session_state.chat_history_loaded = False
# ------------------------
# Initialize FAISS index
# ------------------------
@st.cache_resource
def init_index():
    """Initialize FAISS index with document chunks."""
    all_chunks = load_documents_from_folder(DATA_FOLDER)
    if not all_chunks:
        print("⚠️ No valid documents found in data folder")
        return None, None, []
    index, metas = create_or_load_index(all_chunks, rebuild=False)
    file_chunk_count = {}
    for chunk in metas:
        metadata = chunk.get("metadata", {})
        src = metadata.get("source_file", "Unknown")
        file_chunk_count[src] = file_chunk_count.get(src, 0) + 1
    print("✅ Files loaded with chunk counts:")
    for f, count in file_chunk_count.items():
        print(f" {f}: {count} chunks")
    print(f"✅ Index created with {len(metas)} total chunks")
    return index, metas, all_chunks
index, metas, all_chunks = init_index()
st.session_state.index = index
st.session_state.metas = metas
st.session_state.all_chunks = all_chunks
if metas:
    load_domain_keywords(metas)
# ------------------------
# Helper Functions
# ------------------------
def _get_core_vars():
    """Safely fetch core variables from session state and config."""
    idx = st.session_state.get("index")
    metas_local = st.session_state.get("metas")
    api_key = API_KEY
    model_name = MODEL_NAME
    return idx, metas_local, api_key, model_name, DOMAIN_KEYWORDS
def format_response_with_proper_lists(text: str) -> str:
    """Format response for perfect rendering."""
    if not text or not text.strip():
        return text
   
    lines = text.split('\n')
    formatted_lines = []
   
    for i, line in enumerate(lines):
        stripped = line.strip()
       
        if not stripped:
            if formatted_lines and formatted_lines[-1] != '':
                formatted_lines.append('')
            continue
       
        step_match = re.match(r'^(?:Step\s+)?(\d+)[:\.\-]\s*(.+)$', stripped, re.IGNORECASE)
        if step_match:
            step_num = step_match.group(1)
            step_text = step_match.group(2).strip()
           
            if formatted_lines and formatted_lines[-1] != '':
                formatted_lines.append('')
           
            formatted_lines.append(f'**Step {step_num}: {step_text}**')
            continue
       
        bullet_match = re.match(r'^[•\-\*]\s+(.+)$', stripped)
        if bullet_match:
            bullet_text = bullet_match.group(1).strip()
           
            if formatted_lines and formatted_lines[-1] != '' and not formatted_lines[-1].startswith('•'):
                formatted_lines.append('')
           
            formatted_lines.append(f'• {bullet_text}')
            continue
       
        sub_bullet_match = re.match(r'^\s{2,}[•\-\*]\s+(.+)$', line)
        if sub_bullet_match:
            sub_text = sub_bullet_match.group(1).strip()
            formatted_lines.append(f' - {sub_text}')
            continue
       
        formatted_lines.append(stripped)
   
    result = '\n'.join(formatted_lines)
   
    while '\n\n\n' in result:
        result = result.replace('\n\n\n', '\n\n')
   
    return result.strip()
def render_mermaid_diagram(mermaid_payload: dict):
    """
    Render Mermaid diagrams using st.components - FULL SCREEN VERSION
    """
    if not mermaid_payload:
        return
   
    import logging
    logger = logging.getLogger(__name__)
   
    logger.info(f"🎨 Rendering {mermaid_payload.get('visualization_type')} diagram")
   
    # Extract data from payload
    mermaid_code = mermaid_payload.get('content', '')
    viz_type = mermaid_payload.get('visualization_type', 'diagram')
    title = mermaid_payload.get('title', 'Process Diagram')
   
    if not mermaid_code:
        logger.warning("⚠️ No mermaid code to render")
        return
   
    # Display title
    st.markdown(f"### 📊 {title}")
   
    # Create HTML file content that will render Mermaid - FULL SCREEN NO BOX
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: #ffffff;
                font-family: Arial, sans-serif;
                overflow: auto;
            }}
            .mermaid {{
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                padding: 40px 20px;
                background: #ffffff;
            }}
            .mermaid svg {{
                max-width: 100%;
                height: auto;
            }}
        </style>
    </head>
    <body>
        <div class="mermaid">
    {mermaid_code}
        </div>
        <script>
            mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
            mermaid.contentLoaded();
        </script>
    </body>
    </html>
    """
   
    # Render using st.components with FULL HEIGHT
    components.html(html_content, height=800, scrolling=True)
   
    logger.info(f"✅ Diagram rendered: {viz_type}")
# ------------------------
# MODERN SIDEBAR
# ------------------------
with st.sidebar:
    st.markdown("### ☰ DASHBOARD")
    if "collapse_system" not in st.session_state:
        st.session_state.collapse_system = False
    if "collapse_languages" not in st.session_state:
        st.session_state.collapse_languages = False
    if "collapse_knowledge" not in st.session_state:
        st.session_state.collapse_knowledge = False
    if "collapse_sources" not in st.session_state:
        st.session_state.collapse_sources = False
    def toggle_section(section_name):
        for sec in ["collapse_system", "collapse_languages", "collapse_knowledge", "collapse_sources"]:
            st.session_state[sec] = (sec == section_name) and not st.session_state[sec]
    # SYSTEM STATUS
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    arrow_system = "▼" if st.session_state.collapse_system else "▶"
    if st.button(f"{arrow_system} ⚙️ SYSTEM STATUS", key="toggle_system"):
        toggle_section("collapse_system")
        st.rerun()
    if st.session_state.collapse_system:
        try:
            if not metas or index.ntotal == 0:
                st.markdown("❌ RAG pipeline Not loaded")
            else:
                st.markdown("✅ RAG pipeline Working")
            if API_KEY:
                st.markdown("✅ Gemini API Key Configured")
                st.markdown(f"🤖 Model: `{MODEL_NAME}`")
            else:
                st.markdown("❌ Gemini API Key Missing")
        except Exception as e:
            st.markdown(f"❌ System Error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
    # LANGUAGE SUPPORT
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    arrow_languages = "▼" if st.session_state.collapse_languages else "▶"
    if st.button(f"{arrow_languages} 🌐 LANGUAGE SUPPORT", key="toggle_languages"):
        toggle_section("collapse_languages")
        st.rerun()
    if st.session_state.collapse_languages:
        st.markdown("✅ **Supported Languages:** English & Swahili")
        st.markdown(f"🌍 **Active:** `{', '.join(ALLOWED_LANGUAGES)}`")
    st.markdown('</div>', unsafe_allow_html=True)
    # KNOWLEDGE BASE
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    arrow_knowledge = "▼" if st.session_state.collapse_knowledge else "▶"
    if st.button(f"{arrow_knowledge} 📚 KNOWLEDGE BASE", key="toggle_knowledge"):
        toggle_section("collapse_knowledge")
        st.rerun()
    if st.session_state.collapse_knowledge:
        if metas:
            st.markdown(f"**Total Chunks:** {len(metas)} ✅")
            if st.button("📋 View Chunks", key="chunk_preview_btn"):
                st.session_state.active_tab = "CHUNK PREVIEW"
                st.rerun()
        else:
            st.markdown("❌ No chunks found")
    st.markdown('</div>', unsafe_allow_html=True)
    # SOURCE FILES
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    arrow_sources = "▼" if st.session_state.collapse_sources else "▶"
    if st.button(f"{arrow_sources} 📁 SOURCE FILES", key="toggle_sources"):
        toggle_section("collapse_sources")
        st.rerun()
    if st.session_state.collapse_sources:
        if DOCS_PATHS:
            st.markdown(f"✅ **Total Files:** {len(DOCS_PATHS)}")
            st.markdown("---")
            for path in DOCS_PATHS[:5]:
                st.markdown(f"📄 {os.path.basename(path)}")
            if len(DOCS_PATHS) > 5:
                st.markdown(f"... and {len(DOCS_PATHS) - 5} more")
        else:
            st.markdown("❌ No files found")
    st.markdown('</div>', unsafe_allow_html=True)
    # ------------------------
    # USER LOGIN SECTION
    # ------------------------
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
   
    if st.session_state.user_email:
        # User is logged in
        st.markdown("### 👤 User Profile")
        st.success(f"✅ Logged in as:\n{st.session_state.user_email}")
       
        # Show conversation stats
        summary = chat_history_manager.get_conversation_summary(st.session_state.user_email)
        st.info(f"📊 **Your Stats:**\n- {summary['total_conversations']} conversations\n- {summary['total_messages']} messages")
       
        # Action buttons
        col1, col2 = st.columns(2)
       
        with col1:
            if st.button("🔄 New Chat", key="new_chat_btn", use_container_width=True):
                chat_history_manager.start_new_conversation(st.session_state.user_email)
                st.session_state.messages = []
                st.session_state.chat_history_loaded = False
                st.success("✅ New conversation started!")
                st.rerun()
       
        with col2:
            if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
                st.session_state.user_email = None
                st.session_state.messages = []
                st.session_state.chat_history_loaded = False
                st.rerun()
       
        # View History Button - NEW
        if st.button("📜 View History", key="view_history_btn", use_container_width=True):
            st.session_state.active_tab = "CHAT HISTORY"
            st.rerun()
       
        # Optional: Clear all history button
        if st.button("🗑️ Clear All History", key="clear_history_btn", use_container_width=True, type="secondary"):
            if "confirm_clear" not in st.session_state:
                st.session_state.confirm_clear = False
           
            if not st.session_state.confirm_clear:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Click again to confirm deletion of ALL your chat history!")
            else:
                chat_history_manager.clear_history(st.session_state.user_email)
                st.session_state.messages = []
                st.session_state.chat_history_loaded = False
                st.session_state.confirm_clear = False
                st.success("✅ All history cleared!")
                st.rerun()
   
    else:
        # User is not logged in
        st.markdown("### 👤 User Login")
        st.info("👋 Please enter your email to start chatting and save your conversation history")
       
        email_input = st.text_input(
            "Email Address",
            placeholder="your.email@example.com",
            key="email_input_field",
            label_visibility="collapsed"
        )
       
        if st.button("🔐 Login", key="login_btn", use_container_width=True):
            if email_input and "@" in email_input and "." in email_input:
                st.session_state.user_email = email_input.strip().lower()
                st.session_state.chat_history_loaded = False
                st.success(f"✅ Welcome, {email_input}!")
                st.rerun()
            else:
                st.error("❌ Please enter a valid email address")
   
    st.markdown('</div>', unsafe_allow_html=True)
# ------------------------
# FULL VISUALIZATION PAGE
# ------------------------
def render_full_visualization_page():
    """
    Render full-screen visualization page - CLEAN VERSION
    Only shows: Title, Back Button, Info Card, Diagram, and Controls
    """
   
    # Check if we have visualization data
    if not st.session_state.current_visualization_data:
        st.error("❌ No visualization data available")
        # Use unique key with timestamp/random
        if st.button("⬅️ Back to Chat", key=f"back_error_{id(st.session_state)}"):
            st.session_state.show_full_visualization = False
            st.rerun()
        return
   
    viz_data = st.session_state.current_visualization_data
   
    # Full-screen header with gradient
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 40px 20px; text-align: center; border-bottom: 4px solid #5a67d8;
                    box-shadow: 0 6px 20px rgba(0,0,0,0.15); margin: -80px -20px 40px -20px;">
            <h1 style="font-size: 42px; color: #ffffff; margin: 0; font-weight: 700;
                       text-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                📊 {viz_data.get('title', 'Process Visualization')}
            </h1>
          
        </div>
        """,
        unsafe_allow_html=True
    )
   
    # Only Back to Chat button - centered - UNIQUE KEY
    col1, col2, col3 = st.columns([4, 2, 4])
   
    with col2:
        # Use unique key that won't conflict
        if st.button("⬅️ Back to Chat", key=f"back_viz_{hash(str(viz_data))}", use_container_width=True):
            st.session_state.show_full_visualization = False
            st.session_state.current_visualization_data = None
            st.rerun()
   
    st.markdown("<br>", unsafe_allow_html=True)
   
# 🔥 Show summarization notice if applicable
    was_summarized = viz_data.get('was_summarized', False)
    if was_summarized:
        original_count = viz_data.get('original_step_count', 0)
        phase_count = viz_data.get('step_count', 0)
        st.info(
            f"ℹ️ **Simplified View**: This diagram shows {phase_count} main phases. "
            f"The original process had {original_count} detailed steps. "
            f"The full answer in chat contains all details."
        )
   
    # Main diagram section (SINGLE RENDER ONLY)
    st.markdown(
        """
        <h2 style="color: #2d3748; font-size: 28px; font-weight: 700; margin-bottom: 20px; text-align: center;">
            🎨 Interactive Diagram
        </h2>
        """,
        unsafe_allow_html=True
    )
   
    # Render the actual Mermaid diagram - FULL SCREEN (ONLY ONCE)
    render_mermaid_diagram(viz_data)
   
    st.markdown("<br><br>", unsafe_allow_html=True)
   
    # Control buttons row (MOVED AFTER DIAGRAM)
   
# ------------------------
# ------------------------
# MAIN CHATBOT INTERFACE
# ------------------------
# ------------------------
# CHAT HISTORY LOADER
# ------------------------
def load_user_chat_history():
    """Load user's previous conversation history on login"""
    if not st.session_state.user_email:
        return
   
    if st.session_state.chat_history_loaded:
        return
   
    try:
        # Get recent messages for context
        context_limit = int(os.getenv("CHAT_HISTORY_CONTEXT_LIMIT", "10"))
        recent_messages = chat_history_manager.get_recent_messages(
            st.session_state.user_email,
            limit=context_limit
        )
       
        if recent_messages:
            # Load into session state
            st.session_state.messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in recent_messages
            ]
            print(f"📂 Loaded {len(recent_messages)} messages for {st.session_state.user_email}")
        else:
            print(f"📭 No previous messages for {st.session_state.user_email}")
       
        st.session_state.chat_history_loaded = True
       
    except Exception as e:
        print(f"⚠️ Error loading chat history: {e}")
        st.session_state.chat_history_loaded = True
# ------------------------
# MAIN CHATBOT INTERFACE
# ------------------------
def render_chatbot_tab():
    """Modern chatbot interface with FIXED bottom input bar and Mermaid diagram rendering."""
    # Load user's chat history if logged in
    if st.session_state.user_email and not st.session_state.chat_history_loaded:
        load_user_chat_history()
    # Initialize mermaid payload in session state
    if "mermaid_payload" not in st.session_state:
        st.session_state.mermaid_payload = None
    # Header
    st.markdown(
        """
        <div class="chat-header">
            <h1>💬 AI Chatbot Assistant</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Chat display wrapper
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
   
    # EMPTY STATE
    if len(st.session_state.messages) == 0:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">🤖</div>
                <h2>What's on your mind today?</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Display messages with avatars
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div class="message-row">
                        <div class="message-avatar user-avatar">👤</div>
                        <div class="message-content user-message">{msg["content"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="message-row">
                        <div class="message-avatar assistant-avatar">🤖</div>
                        <div class="message-content assistant-message">{msg["content"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
               
                # TTS and Visualization buttons - FOR EVERY ASSISTANT MESSAGE
                col0, col1, col2, col3 = st.columns([1.7, 1, 1, 6.3]) # Total still ~10 - positions unchanged
                with col0:
                    st.empty() # Invisible padding space
                with col1:
                    tts_key = f"tts_hist_{i}"
                    # Enhanced: Icon-only for compactness, full tooltip, disable if no content
                    if st.button("🔊", key=tts_key, help="Play audio response (TTS)", disabled=not msg["content"].strip()):
                        clean_text = re.sub(r'<[^>]+>', '', msg["content"]).strip()
                        if clean_text:
                            play_tts(clean_text)
                        else:
                            st.toast("No content to read aloud", icon="ℹ️")
                with col2:
                    # Viz button stays the same position, enhanced: conditional show/hide, better tooltip
                    msg_mermaid = msg.get("mermaid_payload")
                    if msg_mermaid:
                        viz_btn_key = f"viz_btn_{i}"
                        if st.button("📊", key=viz_btn_key, help="Open interactive diagram in full view", disabled=not msg_mermaid):
                            st.session_state.current_visualization_data = msg_mermaid
                            st.session_state.show_full_visualization = True
                            st.session_state.awaiting_visualization = False
                            st.rerun()
                    else:
                        st.empty() # Invisible placeholder to maintain layout when no viz
                with col3:
                    st.empty() # Remaining space
# Loading state
    if st.session_state.processing:
        st.markdown(
            """
            <div class="message-row" id="loading-message">
                <div class="message-avatar assistant-avatar">🤖</div>
                <div class="message-content assistant-message">
                    <div class="loading-dots">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
       
        # AGGRESSIVE SCROLL - Keep trying until loading message is visible
        components.html(
            """
            <script>
                function scrollToLoadingMessage() {
                    const parentDoc = window.parent.document;
                    const loading = parentDoc.querySelector('#loading-message');
                   
                    if (loading) {
                        // Scroll the loading message into view
                        loading.scrollIntoView({behavior: 'smooth', block: 'end'});
                       
                        // Extra scroll padding to push it further up
                        setTimeout(() => {
                            window.parent.scrollBy(0, 150);
                        }, 200);
                       
                        console.log('✅ Scrolled to loading message');
                        return true;
                    }
                    console.log('⏳ Loading message not found yet');
                    return false;
                }
               
                // Multiple attempts with increasing delays to catch DOM updates
                let attempts = 0;
                const maxAttempts = 15;
               
                function retryScroll() {
                    if (attempts < maxAttempts) {
                        if (!scrollToLoadingMessage()) {
                            attempts++;
                            setTimeout(retryScroll, 100 + (attempts * 50));
                        } else {
                            console.log('✅ Successfully scrolled on attempt ' + attempts);
                        }
                    }
                }
               
                // Start immediately and keep retrying
                retryScroll();
            </script>
            """,
            height=0
        )
    # 🔥 MOVED: Process message (runs after loading renders, during the same execution)
    if st.session_state.processing and len(st.session_state.messages) > 0:
        if st.session_state.messages[-1]["role"] == "user":
            final_input = st.session_state.messages[-1]["content"]
           
            # 🔥 PRIORITY 1: Check if we're in clarification mode (after user said "yes")
            if st.session_state.awaiting_followup_type == "clarify":
                user_input_lower = final_input.lower().strip()
               
                # Check if user is clarifying their choice
                is_visual_choice = any(word in user_input_lower for word in ["visual", "diagram", "chart", "graph"])
                is_followup_choice = any(word in user_input_lower for word in ["follow", "question", "ask"])
               
                if is_visual_choice:
                    # Generate visual using saved context
                    original_query = st.session_state.get('last_query_for_visuals', '')
                    original_response = st.session_state.get('last_rag_response', '')
                   
                    if not original_query or not original_response:
                        error_msg = "I couldn't find the original question. Please ask your question again."
                        st.session_state.messages.append({"role": "assistant", "content": error_msg, "mermaid_payload": None})
                        st.session_state.processing = False
                        st.session_state.awaiting_followup_type = None
                        st.session_state.awaiting_followup = None
                        st.session_state.awaiting_visualization = False
                        st.rerun()
                   
                    print(f"✅ Generating visual for saved query: {original_query[:50]}")
                   
                    mermaid_payload = process_visual_request(
                        query=original_query,
                        rag_response=original_response,
                        language=st.session_state.last_input_language or "en",
                        force_generate=True
                    )
                   
                    if mermaid_payload:
                       
                        # Clean up clarification messages
                        messages_to_remove = []
                        for i in range(len(st.session_state.messages) - 1, -1, -1):
                            msg = st.session_state.messages[i]
                            if msg["role"] == "user" and msg["content"].lower().strip() in ["yes", "visuals"]:
                                messages_to_remove.append(i)
                            elif msg["role"] == "assistant" and "Yes for what" in msg["content"]:
                                messages_to_remove.append(i)
                            if len(messages_to_remove) >= 3:
                                break
                       
                        for idx in sorted(messages_to_remove, reverse=True):
                            st.session_state.messages.pop(idx)
                       
                        # Append new message with visual payload
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "📊 Here's the interactive diagram for your query:",
                            "mermaid_payload": mermaid_payload
                        })
                       
                        # Clear states
                        st.session_state.awaiting_visualization = False
                        st.session_state.awaiting_followup = None
                        st.session_state.awaiting_followup_type = None
                        st.session_state.processing = False
                        st.session_state.mermaid_payload = None
                        print("✅ Visual generated from saved context, conversation cleaned")
                        st.rerun()
                    else:
                        error_msg = "Sorry, I couldn't generate a visual for that response."
                        st.session_state.messages.append({"role": "assistant", "content": error_msg, "mermaid_payload": None})
                        st.session_state.processing = False
                        st.session_state.awaiting_followup_type = None
                        st.rerun()
                       
                elif is_followup_choice:
                    # User wants follow-up question
                    final_input = st.session_state.awaiting_followup
                    st.session_state.messages[-1]["content"] = f"🔄 {final_input}"
                    st.session_state.awaiting_followup = None
                    st.session_state.awaiting_visualization = False
                    st.session_state.awaiting_followup_type = None
                    # Continue to RAG processing below
                   
                else:
                    # User input unclear
                    clarify_msg = "Please specify: type 'visuals' to see the diagram, or 'follow-up' to explore the suggested question."
                    st.session_state.messages.append({"role": "assistant", "content": clarify_msg, "mermaid_payload": None})
                    st.session_state.processing = False
                    if len(st.session_state.messages) >= 2 and st.session_state.messages[-2]["role"] == "user":
                        st.session_state.messages.pop(-2)
                    st.rerun()
           
            # 🔥 PRIORITY 2: Check for YES/NO (only if NOT in clarify mode)
            elif (st.session_state.awaiting_followup or st.session_state.awaiting_visualization):
                confirmation_words = ["yes", "sure", "ok", "okay", "yeah", "yep", "yup",
                                    "please", "go ahead", "tell me", "explain", "show me"]
                denial_words = ["no", "nope", "nah", "cancel", "skip", "no thanks", "not now"]
               
                user_input_lower = final_input.lower().strip()
                is_yes = user_input_lower in confirmation_words or any(word in user_input_lower for word in confirmation_words)
                is_no = user_input_lower in denial_words or any(word in user_input_lower for word in denial_words)
               
                if is_yes:
                    # User said YES - ask for clarification
                    st.session_state.awaiting_followup_type = "clarify"
                    clarify_msg = "Yes for what? (following questions / visuals)"
                    st.session_state.messages.append({"role": "assistant", "content": clarify_msg, "mermaid_payload": None})
                    st.session_state.processing = False
                    st.rerun()
               
                elif is_no:
                    # User declined both options
                    st.session_state.awaiting_followup = None
                    st.session_state.awaiting_visualization = False
                    st.session_state.awaiting_followup_type = None
                    st.session_state.processing = False
                    print("❌ User declined both options")
                    st.rerun()
                else:
                    # Not a yes/no - treat as new question
                    print("🔄 Not a confirmation - treating as new question")
                    st.session_state.awaiting_followup = None
                    st.session_state.awaiting_visualization = False
                    st.session_state.awaiting_followup_type = None
                    # Continue to RAG processing below
           
            # 🔥 MAIN RAG PROCESSING - Runs for new questions or after follow-up selection
            if st.session_state.processing:
                print(f"🔄 Starting RAG processing for: {final_input}")
               
                idx, metas_local, api_key, model_name, domain_keywords = _get_core_vars()
                model = genai.GenerativeModel(MODEL_NAME)
                input_lang = detect_language(final_input)
                print(f"🌐 Detected language: {input_lang}")
       
                if input_lang not in ALLOWED_LANGUAGES:
                    print(f"⚠️ Language '{input_lang}' not allowed, defaulting to English")
                    input_lang = "en"
                # STEP 1: DETECT GREETING
                greeting_info = {"is_greeting": False, "has_question": False, "greeting_text": "", "user_name": "", "question_part": ""}
               
                try:
                    greeting_prompt = prompts.get_greeting_detection_prompt(final_input)
                    chat_response = model.generate_content(greeting_prompt)
                    content = getattr(chat_response, "text", None) or "{}"
                   
                    content = content.strip()
                    if content.startswith("```json:disable-run
                        content = content[7:]
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                   
                    try:
                        greeting_info = json.loads(content)
                        print(f"✓ Greeting Detection Result: {greeting_info}")
                    except json.JSONDecodeError:
                        print(f"⚠️ Failed to parse greeting JSON: {content}")
                        greeting_info = {"is_greeting": False, "has_question": True, "greeting_text": "", "user_name": "", "question_part": final_input}
                except Exception as e:
                    print(f"⚠️ Greeting detection error: {e}")
                    greeting_info = {"is_greeting": False, "has_question": True, "greeting_text": "", "user_name": "", "question_part": final_input}
                response_parts = []
                rag_output = ""
                question_en = "" # ✅ ADD THIS
                query_en = "" # ✅ ADD THIS
               
                # STEP 2: ROUTE BASED ON GREETING DETECTION
               
                # CASE 1: PURE GREETING (no question)
                if greeting_info.get("is_greeting") and not greeting_info.get("has_question"):
                    print("✓ Detected: PURE GREETING - No RAG call")
                    greeting_response = prompts.get_pure_greeting_response(
                        user_input=final_input,
                        user_name=greeting_info.get("user_name", "")
                    )
                    if input_lang != "en":
                        greeting_response = translate_text(greeting_response, target_lang=input_lang)
                    response_parts.append(greeting_response)
                   
                    complete_response = "".join(response_parts)
                    st.session_state.messages.append({"role": "assistant", "content": complete_response, "mermaid_payload": None})
                    st.session_state.processing = False
                    st.session_state.mermaid_payload = None
                    st.rerun()
               
                            # CASE 2: GREETING + QUESTION
                elif greeting_info.get("is_greeting") and greeting_info.get("has_question"):
                    print("✓ Detected: GREETING + QUESTION - Will greet then answer")
                    greeting_response = prompts.get_pure_greeting_response(
                        user_input=greeting_info.get("greeting_text", "hello"),
                        user_name=greeting_info.get("user_name", "")
                    )
                    if input_lang != "en":
                        greeting_response = translate_text(greeting_response, target_lang=input_lang)
                    response_parts.append(greeting_response)
                    response_parts.append("\n\n")
                   
                    question_text = greeting_info.get("question_part", final_input)
                    question_en = question_text if input_lang == "en" else translate_text(question_text, target_lang="en")
                    question_en = fuzzy_match_text(question_en, domain_keywords)
                   
                    try:
                        rag_output = rag_answer(question_en, idx, metas_local, api_key, model_name)
                        if input_lang != "en":
                            rag_output = translate_text(rag_output, target_lang=input_lang)
                        response_parts.append(rag_output)
                       
                    except Exception as e:
                        error_msg = f"⚠️ Error: {e}"
                        if input_lang != "en":
                            error_msg = translate_text(error_msg, target_lang=input_lang)
                        response_parts.append(error_msg)
                # CASE 3: QUESTION ONLY
                else:
                    print("✓ Detected: QUESTION ONLY - Direct RAG call")
                    query_en = final_input if input_lang == "en" else translate_text(final_input, target_lang="en")
                    query_en = fuzzy_match_text(query_en, domain_keywords)
                   
                    try:
                        rag_output = rag_answer(query_en, idx, metas_local, api_key, model_name)
                        if input_lang != "en":
                            rag_output = translate_text(rag_output, target_lang=input_lang)
                        response_parts.append(rag_output)
                       
                        # ✅ SAVE CONTEXT IMMEDIATELY after successful RAG
                        st.session_state.last_rag_response = rag_output
                        st.session_state.last_query_for_visuals = query_en
                        st.session_state.last_input_language = input_lang
                        print(f"✅ Context saved: query='{query_en[:50]}...', response_len={len(rag_output)}")
                       
                    except Exception as e:
                        error_msg = f"⚠️ Error: {e}"
                        if input_lang != "en":
                            error_msg = translate_text(error_msg, target_lang=input_lang)
                        response_parts.append(error_msg)
                # ✅ Save context IMMEDIATELY after each RAG call
                if rag_output:
                    # Use the actual query that was used for RAG
                    actual_query = question_en if greeting_info.get("has_question") else query_en
                   
                    st.session_state.last_rag_response = rag_output
                    st.session_state.last_query_for_visuals = actual_query if actual_query else final_input
                    st.session_state.last_input_language = input_lang
                   
                    print(f"✅ SAVED CONTEXT - Query: '{st.session_state.last_query_for_visuals[:50]}...'")
                    print(f"✅ SAVED CONTEXT - Response length: {len(st.session_state.last_rag_response)}")
                # STEP 3: FORMAT RESPONSE
                raw_response = "".join(response_parts)
                formatted_response = format_response_with_proper_lists(raw_response)
                # STEP 4: GENERATE FOLLOW-UP QUESTIONS
                followup_text = ""
                visual_prompt = ""
                if rag_output and (greeting_info.get("has_question") or not greeting_info.get("is_greeting")):
                    try:
                        followup_prompt = prompts.get_followup_generation_prompt(rag_output)
                        followup_resp = model.generate_content(followup_prompt)
                        followup_text = getattr(followup_resp, "text", "").strip()
                        if followup_text and input_lang != "en":
                            followup_text = translate_text(followup_text, target_lang=input_lang)
                        if followup_text:
                            st.session_state.awaiting_followup = followup_text
                       
                        # Generate visual suggestion
                        visual_prompt = "Do you want to see visuals for this answer?"
                        if input_lang != "en":
                            visual_prompt = translate_text(visual_prompt, target_lang=input_lang)
                    except Exception as e:
                        print(f"⚠️ Follow-up generation error: {e}")
                # STEP 5: BUILD COMPLETE RESPONSE
                complete_response = formatted_response
               
                if followup_text:
                    followup_label = "💡 If you want, I can also tell you about:" if input_lang == "en" else translate_text("💡 If you want, I can also tell you about:", target_lang=input_lang)
                    complete_response += f'\n\n**{followup_label}** 👉 {followup_text}'
               
                # Set state to await user response
                st.session_state.awaiting_visualization = True
                st.session_state.awaiting_followup = followup_text
               
                # Add visual suggestion
                if visual_prompt:
                    visual_label = "💡 "
                    complete_response += f'\n\n**{visual_label}{visual_prompt}**'
                # STEP 6: GENERATE VISUALIZATION - DISABLED (Only generate when user explicitly asks)
                mermaid_payload = None # Don't generate visual automatically
                st.session_state.mermaid_payload = None # Clear any previous visuals
                # STEP 7: SAVE AND DISPLAY
                st.session_state.messages.append({"role": "assistant", "content": complete_response})
               
                # Save to history if user is logged in
                if st.session_state.user_email:
                    chat_history_manager.add_message(
                        email=st.session_state.user_email,
                        role="assistant",
                        content=complete_response,
                        metadata={
                            "language": input_lang,
                            "had_visualization": bool(st.session_state.mermaid_payload),
                            "had_followup": bool(followup_text)
                        }
                    )
               
                st.session_state.processing = False
                st.rerun()
    # 🔥 AUTO-SCROLL ANCHOR - Place at bottom inside wrapper (only for non-loading messages)
    if len(st.session_state.messages) > 0 and not st.session_state.processing:
        st.markdown('<div id="scroll-anchor"></div>', unsafe_allow_html=True)
        components.html(
            """
            <script>
                function scrollToAnchor() {
                    const parentDoc = window.parent.document;
                    const anchor = parentDoc.querySelector('#scroll-anchor');
                   
                    if (anchor) {
                        anchor.scrollIntoView({behavior: 'smooth', block: 'end'});
                        return true;
                    }
                    return false;
                }
               
                scrollToAnchor();
                setTimeout(scrollToAnchor, 50);
                setTimeout(scrollToAnchor, 150);
            </script>
            """,
            height=0
        )
    st.markdown('</div>', unsafe_allow_html=True)
   
    # Fixed input bar at bottom
    st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([8, 1, 1])
    with col1:
        st.markdown('<div class="input-text-wrapper">', unsafe_allow_html=True)
       
        # 🔥 Show waveform INSIDE input when recording
        if st.session_state.listening:
            # Recording waveform in input box
            components.html(
                """
                <div style="width: 100%; height: 50px; background: #f0f9ff; border: 2px solid #3B82F6;
                            border-radius: 12px; display: flex; align-items: center; padding: 0 15px; overflow: hidden;">
                    <canvas id="miniWaveform" style="flex: 1; height: 100%; width: 100%;"></canvas>
                </div>
                <style>
                    @keyframes recordPulse {
                        0%, 100% { opacity: 1; transform: scale(1); }
                        50% { opacity: 0.4; transform: scale(1.3); }
                    }
                </style>
                <script>
                    const canvas = document.getElementById('miniWaveform');
                    const ctx = canvas.getContext('2d');
                    let audioContext, analyser, microphone, dataArray;
                    let time = 0; // For subtle flow
                   
                    canvas.width = canvas.offsetWidth;
                    canvas.height = canvas.offsetHeight;
                   
                    async function startMiniWaveform() {
                        try {
                            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                            audioContext = new (window.AudioContext || window.webkitAudioContext)();
                            analyser = audioContext.createAnalyser();
                            microphone = audioContext.createMediaStreamSource(stream);
                            analyser.fftSize = 1024; // Higher resolution for more realistic detail
                            analyser.smoothingTimeConstant = 0.85; // Smoother for realistic feel
                            dataArray = new Uint8Array(analyser.frequencyBinCount);
                            microphone.connect(analyser);
                            drawMini();
                        } catch (err) {
                            console.error('Mic error:', err);
                            // Fallback to simulated data if mic fails
                            drawMini();
                        }
                    }
                   
                    function drawMini() {
                        requestAnimationFrame(drawMini);
                       
                        // Use mic data or fallback simulation
                        if (analyser) {
                            analyser.getByteTimeDomainData(dataArray);
                        } else {
                            // Simulated realistic waveform for demo
                            for (let i = 0; i < dataArray.length; i++) {
                                dataArray[i] = 128 + Math.sin(i * 0.05 + time * 0.02) * 50 + Math.sin(i * 0.1 + time * 0.01) * 30;
                            }
                        }
                       
                        // Fill entire canvas with the light background color (no blue)
                        ctx.fillStyle = '#f0f9ff';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                       
                        const sliceWidth = canvas.width / dataArray.length;
                        let x = 0;
                        const centerY = canvas.height / 2;
                       
                        // Draw filled waveform path for realistic filled look
                        ctx.beginPath();
                        ctx.moveTo(0, centerY);
                       
                        for (let i = 0; i < dataArray.length; i++) {
                            const v = (dataArray[i] / 128.0) - 1; // -1 to 1 range
                            const amplitude = (canvas.height / 2) * 0.95; // Nearly full height for coverage
                            let y = centerY - (v * amplitude); // Upside down for standard wave (positive up)
                            const subtleRipple = Math.sin((i * 0.03) + time * 0.03) * 2; // Subtle realistic ripple
                            y += subtleRipple;
                           
                            ctx.lineTo(x, y);
                            x += sliceWidth;
                        }
                       
                        ctx.lineTo(canvas.width, centerY);
                        ctx.closePath();
                       
                        // Fill the waveform area with semi-transparent blue for realistic depth
                        const waveGradient = ctx.createLinearGradient(0, centerY, 0, canvas.height);
                        waveGradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)'); // Lighter blue fill
                        waveGradient.addColorStop(1, 'rgba(59, 130, 246, 0.7)'); // Stronger blue at base
                        ctx.fillStyle = waveGradient;
                        ctx.fill();
                       
                        // Outline the wave for crisp, realistic edges
                        ctx.lineWidth = 1.5;
                        ctx.strokeStyle = '#3B82F6';
                        ctx.stroke();
                       
                        time += 1;
                    }
                   
                    startMiniWaveform();
                </script>
                """,
                height=54
            )
        else:
            # Normal text input when not recording
            user_input = st.text_input(
                "Message",
                key=f"text_input_{st.session_state.input_key}",
                value=st.session_state.user_input_text,
                placeholder="Type your message here...",
                label_visibility="collapsed"
            )
       
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="mic-btn-wrapper">', unsafe_allow_html=True)
        if st.session_state.listening:
            # Cancel button when recording
            cancel_clicked = st.button("❌", key=f"cancel_{st.session_state.input_key}", help="Cancel recording")
        else:
            # Mic button when not recording
            mic_clicked = st.button("🎤", key=f"mic_{st.session_state.input_key}", help="Voice input")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="send-btn-wrapper">', unsafe_allow_html=True)
        if st.session_state.listening:
            # Stop button when recording
            stop_clicked = st.button("✅", key=f"stop_{st.session_state.input_key}", help="Stop & send")
        else:
            # Send button when not recording
            send_clicked = st.button("➤", key=f"send_{st.session_state.input_key}", help="Send message")
        st.markdown('</div>', unsafe_allow_html=True)
   
   
    # Handle mic button click
    # Handle cancel button (when recording)
    if st.session_state.listening and 'cancel_clicked' in locals() and cancel_clicked:
        st.session_state.mic_recorder.stop_listening()
        st.session_state.mic_recorder.audio_data.clear()
        st.session_state.listening = False
        st.rerun()
    # Handle stop button (when recording)
    if st.session_state.listening and 'stop_clicked' in locals() and stop_clicked:
        st.session_state.mic_recorder.stop_listening()
        transcript = st.session_state.mic_recorder.transcribe()
        st.session_state.listening = False
       
        if transcript:
            # Store transcript in session state to show in input box
            st.session_state.user_input_text = transcript
        st.rerun()
    # Handle mic button click (when not recording)
    if not st.session_state.listening and 'mic_clicked' in locals() and mic_clicked:
        st.session_state.mic_recorder.start_listening()
        st.session_state.listening = True
        st.rerun()
        # Handle send button click OR Enter key press (when not recording)
    should_send = False
    # Check if send button was clicked
    if not st.session_state.listening and 'send_clicked' in locals() and send_clicked:
        should_send = True
    # Check if Enter was pressed (detect if user_input changed from previous value)
    if not st.session_state.listening and 'user_input' in locals() and user_input and user_input.strip():
        # If current input differs from stored text, it means Enter was pressed
        if user_input.strip() != st.session_state.user_input_text:
            should_send = True
    # Process the message
    if should_send and 'user_input' in locals() and user_input and user_input.strip():
        # Add user message to session
        user_message = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": user_message})
       
        # Save to history if user is logged in
        if st.session_state.user_email:
            chat_history_manager.add_message(
                email=st.session_state.user_email,
                role="user",
                content=user_message,
                metadata={"timestamp": "user_input"}
            )
       
        st.session_state.processing = True
        st.session_state.user_input_text = "" # Clear input
        st.session_state.input_key += 1 # 🔥 ADD THIS LINE - Force input refresh
        st.rerun()
# ------------------------
# Other Tabs
# ------------------------
def render_system_status_tab():
    st.markdown(
        """
        <div class="chat-header">
            <h1>⚙️ System Status</h1>
            <p>Monitor system health and configuration</p>
        </div>
        """,
        unsafe_allow_html=True
    )
   
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
   
    with st.expander("🤖 RAG MODEL", expanded=True):
        try:
            if not metas or index.ntotal == 0:
                st.error("❌ RAG index empty")
            else:
                test_vec = np.random.rand(1, index.d).astype("float32")
                index.search(test_vec, k=1)
                st.success("✅ RAG pipeline working")
        except Exception as e:
            st.error(f"❌ RAG error: {e}")
    with st.expander("🧠 AI MODEL"):
        if API_KEY:
            st.success(f"✅ Gemini AI Model configured: `{MODEL_NAME}`")
            st.info(f"📦 Fallback Model: `{FALLBACK_MODEL}`")
        else:
            st.error("❌ Gemini API key not set")
   
    with st.expander("🌐 LANGUAGE CONFIGURATION"):
        st.success(f"✅ Supported Languages: {', '.join(ALLOWED_LANGUAGES)}")
        st.info("🌐 Language restriction: ONLY English and Swahili")
   
    st.markdown('</div>', unsafe_allow_html=True)
def render_knowledge_base_tab():
    st.markdown(
        """
        <div class="chat-header">
            <h1>📚 Knowledge Base</h1>
            <p>View indexed document statistics</p>
        </div>
        """,
        unsafe_allow_html=True
    )
   
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
   
    if metas:
        st.success("✅ Knowledge Base Loaded Successfully!")
        st.markdown(f"**Total Chunks Indexed:** {len(metas)}")
       
        st.markdown("---")
        st.markdown("### 📊 Statistics")
       
        # Calculate file statistics
        file_chunk_count = {}
        for chunk in metas:
            metadata = chunk.get("metadata", {})
            src = metadata.get("source_file", "Unknown")
            file_chunk_count[src] = file_chunk_count.get(src, 0) + 1
       
        st.markdown(f"**Total Files:** {len(file_chunk_count)}")
        st.markdown(f"**Average Chunks per File:** {len(metas) // max(len(file_chunk_count), 1)}")
    else:
        st.warning("⚠️ No chunks found in the knowledge base.")
   
    st.markdown('</div>', unsafe_allow_html=True)
def render_chunk_preview_tab():
    st.markdown(
        """
        <div class="chat-header">
            <h1>🔎 Chunk Preview</h1>
            <p>View indexed document chunks</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="chat-wrapper chunk-preview-wrapper">', unsafe_allow_html=True)
    if st.button("⬅️ Back to Chat", key="back_to_chat_error"):
        st.session_state.active_tab = "🤖 AI Chatbot"
        st.rerun()
    all_chunks = st.session_state.get("all_chunks", [])
    if not all_chunks:
        st.warning("⚠️ No chunks found in the knowledge base.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    st.markdown(f"**Total Chunks:** {len(all_chunks)}")
    st.markdown("---")
    chunks_per_page = ConfigManager.get('CHUNKS_PER_PAGE', 10, config_type=int)
    total_pages = (len(all_chunks) + chunks_per_page - 1) // chunks_per_page
   
    if "chunk_page" not in st.session_state:
        st.session_state.chunk_page = 0
   
    col1, col2, col3 = st.columns([1, 2, 1])
   
    with col1:
        if st.button("⬅️ Previous", disabled=st.session_state.chunk_page == 0):
            st.session_state.chunk_page -= 1
            st.rerun()
   
    with col2:
        st.markdown(f"<center>Page {st.session_state.chunk_page + 1} of {total_pages}</center>", unsafe_allow_html=True)
   
    with col3:
        if st.button("Next ➡️", disabled=st.session_state.chunk_page >= total_pages - 1):
            st.session_state.chunk_page += 1
            st.rerun()
   
    st.markdown("---")
   
    start_idx = st.session_state.chunk_page * chunks_per_page
    end_idx = min(start_idx + chunks_per_page, len(all_chunks))
   
    for i in range(start_idx, end_idx):
        chunk = all_chunks[i]
        metadata = chunk.get("metadata", {})
        src_file = metadata.get("source_file", "Unknown")
        start_word = metadata.get("start_word", "N/A")
        end_word = metadata.get("end_word", "N/A")
        slide_number = metadata.get("slide_number", None)
        text_preview = chunk.get("text", "")
        preview_snippet = text_preview[:300] + ("..." if len(text_preview) > 300 else "")
        extra_info = f" | Slide {slide_number}" if slide_number else ""
        st.markdown(
            f"""
            <div style="padding:20px; border-radius:12px; background:#f7fafc; margin-bottom:20px; border-left: 5px solid #667eea; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="font-weight:700; color:#667eea; margin-bottom:10px; font-size:16px;">
                    📄 Chunk {i+1} - {src_file}{extra_info}
                </div>
                <div style="color:#718096; font-size:12px; margin-bottom:10px;">
                    Words {start_word}-{end_word}
                </div>
                <div style="background:white; padding:15px; border-radius:8px; font-family:monospace; font-size:14px; line-height:1.8; color:#2d3748;">
                    {preview_snippet}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
   
    st.markdown('</div>', unsafe_allow_html=True)
def render_chat_history_viewer():
    """View all past conversations"""
    st.markdown(
        """
        <div class="chat-header">
            <h1>📜 Chat History</h1>
            <p>Browse through your past conversations</p>
        </div>
        """,
        unsafe_allow_html=True
    )
   
    st.markdown('<div class="chat-wrapper chunk-preview-wrapper">', unsafe_allow_html=True)
   
    # Check if user is logged in
    if not st.session_state.user_email:
        st.warning("⚠️ Please login to view chat history")
        st.info("👉 Enter your email in the sidebar to access your chat history")
       
        if st.button("⬅️ Back to Chat", key="back_to_chat_from_history"):
            st.session_state.active_tab = "🤖 AI Chatbot"
            st.rerun()
       
        st.markdown('</div>', unsafe_allow_html=True)
        return
   
    # Back button
    if st.button("⬅️ Back to Chat", key="back_to_chat_btn"):
        st.session_state.active_tab = "🤖 AI Chatbot"
        st.rerun()
   
    st.markdown("<br>", unsafe_allow_html=True)
   
    # Get all conversations
    try:
        conversations = chat_history_manager.get_all_conversations(st.session_state.user_email)
       
        if not conversations:
            st.info("📭 No chat history found. Start a conversation to build your history!")
            st.markdown('</div>', unsafe_allow_html=True)
            return
       
        # Display summary stats
        summary = chat_history_manager.get_conversation_summary(st.session_state.user_email)
       
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💬 Conversations", summary['total_conversations'])
        with col2:
            st.metric("📨 Messages", summary['total_messages'])
        with col3:
            if summary['last_updated']:
                last_date = summary['last_updated'][:10]
                st.metric("📅 Last Active", last_date)
       
        st.markdown("---")
       
        # Search box
        search_query = st.text_input(
            "🔍 Search in conversations",
            placeholder="Type to search message content...",
            key="history_search_box"
        )
       
        st.markdown("---")
       
        # Filter conversations by search
        filtered_convs = conversations.copy()
        if search_query:
            search_lower = search_query.lower()
            filtered_convs = [
                conv for conv in filtered_convs
                if any(search_lower in msg.get('content', '').lower()
                      for msg in conv.get('messages', []))
            ]
       
        # Show results count
        if search_query:
            st.info(f"🔍 Found {len(filtered_convs)} conversation(s) matching '{search_query}'")
       
        # Display conversations (newest first)
        for i, conv in enumerate(reversed(filtered_convs)):
            session_id = conv.get("session_id", "Unknown")
            started_at = conv.get("started_at", "Unknown")
            messages = conv.get("messages", [])
           
            # Format date nicely
            try:
                date_str = started_at[:10] # YYYY-MM-DD
                time_str = started_at[11:16] # HH:MM
                display_date = f"{date_str} at {time_str}"
            except:
                display_date = started_at
           
            # Get first user message as preview
            preview_text = "No messages"
            for msg in messages:
                if msg.get("role") == "user":
                    preview_text = msg.get("content", "")[:100] + ("..." if len(msg.get("content", "")) > 100 else "")
                    break
           
            # Create expander for each conversation
            with st.expander(
                f"💬 Conversation {len(filtered_convs) - i} • {display_date} • {len(messages)} messages",
                expanded=(i == 0 and not search_query)
            ):
                st.markdown(f"**Preview:** {preview_text}")
                st.markdown("---")
               
                # Display all messages in this conversation
                for msg_idx, msg in enumerate(messages):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    timestamp = msg.get("timestamp", "")
                   
                    # Format timestamp
                    try:
                        time_display = timestamp[11:19] # HH:MM:SS
                    except:
                        time_display = timestamp
                   
                    if role == "user":
                        st.markdown(
                            f"""
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                        color: white; padding: 12px 16px; border-radius: 12px;
                                        margin: 10px 0; border-left: 4px solid #5a67d8;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <strong style="font-size: 14px;">👤 You</strong>
                                    <span style="opacity: 0.8; font-size: 11px;">{time_display}</span>
                                </div>
                                <div style="font-size: 14px; line-height: 1.6;">
                                    {content}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div style="background: #f7fafc; border: 1px solid #e2e8f0;
                                        padding: 12px 16px; border-radius: 12px;
                                        margin: 10px 0; border-left: 4px solid #4FD1C5;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <strong style="font-size: 14px; color: #2d3748;">🤖 Assistant</strong>
                                    <span style="opacity: 0.6; font-size: 11px; color: #718096;">{time_display}</span>
                                </div>
                                <div style="font-size: 14px; line-height: 1.6; color: #2d3748;">
                                    {content}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
               
                # Add action buttons for each conversation
                st.markdown("---")
                col_a, col_b = st.columns(2)
               
                with col_a:
                    if st.button(f"💬 Continue This Chat", key=f"continue_conv_{i}", use_container_width=True):
                        # Load this conversation into main chat
                        st.session_state.messages = [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in messages
                        ]
                        st.session_state.active_tab = "🤖 AI Chatbot"
                        st.success(f"✅ Loaded {len(messages)} messages!")
                        st.rerun()
               
                with col_b:
                    # Show message count
                    st.info(f"📊 {len(messages)} messages in this conversation")
       
        if not filtered_convs:
            st.warning("🔍 No conversations found matching your search")
   
    except Exception as e:
        st.error(f"❌ Error loading chat history: {e}")
        print(f"Error in chat history viewer: {e}")
   
    st.markdown('</div>', unsafe_allow_html=True)
def render_source_files_tab():
    st.markdown(
        """
        <div class="chat-header">
            <h1>📁 Source Files</h1>
            <p>View all loaded document files</p>
        </div>
        """,
        unsafe_allow_html=True
    )
   
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
   
    st.markdown("### Files in Data Folder")
   
    if DOCS_PATHS:
        st.success(f"✅ Total files available: {len(DOCS_PATHS)}")
       
        st.markdown("---")
       
        # Group files by extension
        file_types = {}
        for path in DOCS_PATHS:
            ext = os.path.splitext(path)[1].lower()
            if ext not in file_types:
                file_types[ext] = []
            file_types[ext].append(path)
       
        # Display by type
        for ext, files in sorted(file_types.items()):
            with st.expander(f"**{ext.upper()} Files ({len(files)})**", expanded=True):
                for path in files:
                    file_size = os.path.getsize(path) / 1024 # KB
                    st.markdown(f"📄 **{os.path.basename(path)}** - {file_size:.1f} KB")
    else:
        st.warning("⚠️ No source files found in the data folder")
        st.info(f"📂 Data folder path: `{DATA_FOLDER}`")
   
    st.markdown('</div>', unsafe_allow_html=True)
# ------------------------
# Main Render Logic
# ------------------------
active = st.session_state.active_tab
# 🔥 PRIORITY: Check if full visualization mode is active
if st.session_state.show_full_visualization:
    render_full_visualization_page()
elif active == "🤖 AI Chatbot":
    render_chatbot_tab()
elif active == "SYSTEM STATUS":
    render_system_status_tab()
elif active == "KNOWLEDGE BASE":
    render_knowledge_base_tab()
elif active == "CHUNK PREVIEW":
    render_chunk_preview_tab()
elif active == "SOURCE FILES":
    render_source_files_tab()
elif active == "CHAT HISTORY":
    render_chat_history_viewer()
else:
    render_chatbot_tab()