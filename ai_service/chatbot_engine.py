import os
import pandas as pd
import difflib
import re
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Try to import HuggingFace embeddings, fallback to simple embeddings if fails
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


# -----------------------------
# Simple Fallback Embeddings
# -----------------------------
class SimpleEmbeddings:
    """Simple TF-IDF based embeddings as fallback"""
    
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
        self.is_fitted = False
    
    def embed_documents(self, texts):
        if not self.is_fitted:
            self.vectorizer.fit(texts)
            self.is_fitted = True
        embeddings = self.vectorizer.transform(texts).toarray()
        return embeddings.tolist()
    
    def embed_query(self, text):
        if not self.is_fitted:
            return [0.0] * 384
        embedding = self.vectorizer.transform([text]).toarray()[0]
        return embedding.tolist()


# -----------------------------
# Initialize LLM (Lazy Loading)
# -----------------------------
_llm_instance = None

def get_llm():
    """Lazy load LLM to improve startup time"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY,
            temperature=0.2,  # Lower for more consistent, focused answers
            max_tokens=500,   # Reduced for shorter responses
            streaming=True    # Enable streaming for animation effect
        )
    return _llm_instance


# -----------------------------
# Embedding Model (Lazy Loading with Fallback)
# -----------------------------
_embeddings_instance = None

def get_embeddings():
    """Lazy load embeddings with fallback"""
    global _embeddings_instance
    
    if _embeddings_instance is None:
        try:
            if HUGGINGFACE_AVAILABLE:
                cache_folder = "./models/embeddings"
                os.makedirs(cache_folder, exist_ok=True)
                
                _embeddings_instance = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    cache_folder=cache_folder,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            else:
                raise ImportError("HuggingFace not available")
                
        except Exception as e:
            print(f"⚠️ Using fallback embeddings: {e}")
            _embeddings_instance = SimpleEmbeddings()
    
    return _embeddings_instance


# -----------------------------
# Global Caches & Conversation History
# -----------------------------
vectorstore_cache = {}
dataframe_cache = {}
column_mapping_cache = {}
conversation_context = {}  # Track conversation state per session


# =====================================================
# QUERY CLASSIFICATION
# =====================================================

def classify_query_type(question: str) -> str:
    """Classify query type for appropriate routing"""
    q_lower = question.lower().strip()
    
    # Simple greetings (single response, no repeated intro)
    if q_lower in ['hello', 'hi', 'hey', 'hii', 'hlo', 'hola']:
        return 'simple_greeting'
    
    # Complex greetings or small talk
    greeting_patterns = ['good morning', 'good afternoon', 'good evening', 
                        'how are you', 'what\'s up', 'whats up', 'talk with you',
                        'want to share', 'personal', 'feeling']
    if any(pattern in q_lower for pattern in greeting_patterns):
        return 'small_talk'
    
    # Computational queries
    computation_keywords = ['average', 'mean', 'sum', 'total', 'count', 'how many',
                           'maximum', 'max', 'minimum', 'min', 'calculate', 
                           'median', 'std', 'variance']
    if any(keyword in q_lower for keyword in computation_keywords):
        return 'computation'
    
    # Data-specific queries
    data_keywords = ['survived', 'died', 'passengers', 'show', 'display', 
                    'what is', 'tell me', 'data', 'dataset', 'column']
    if any(keyword in q_lower for keyword in data_keywords):
        return 'data_query'
    
    return 'general'


# =====================================================
# RAG PART
# =====================================================

def build_vectorstore_from_html(report_path: str):
    """Build vectorstore with optimized chunking"""
    if not os.path.exists(report_path):
        return None

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text(separator="\n")
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "]
        )

        docs = splitter.create_documents([text])
        vectorstore = Chroma.from_documents(docs, get_embeddings())
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ Vectorstore error: {e}")
        return None


# =====================================================
# COMPUTATION ENGINE
# =====================================================

def detect_aggregation(question: str) -> Optional[str]:
    """Detect aggregation type"""
    q = question.lower()
    
    patterns = {
        'count': ['count', 'how many', 'number of', 'total number'],
        'sum': ['sum', 'total', 'add up'],
        'mean': ['average', 'mean', 'avg'],
        'max': ['maximum', 'max', 'highest', 'largest'],
        'min': ['minimum', 'min', 'lowest', 'smallest'],
    }
    
    for agg_type, keywords in patterns.items():
        if any(keyword in q for keyword in keywords):
            return agg_type
    
    return None


def detect_column_fuzzy(df: pd.DataFrame, question: str, threshold: float = 0.6) -> Optional[str]:
    """Fuzzy column matching with caching"""
    cache_key = f"{id(df)}_{question}"
    
    if cache_key in column_mapping_cache:
        return column_mapping_cache[cache_key]
    
    words = question.lower().split()
    best_match = None
    best_score = 0

    for col in df.columns:
        col_lower = col.lower()
        
        if col_lower in question.lower():
            column_mapping_cache[cache_key] = col
            return col
        
        for word in words:
            score = difflib.SequenceMatcher(None, word, col_lower).ratio()
            if score > best_score:
                best_score = score
                best_match = col

    result = best_match if best_score >= threshold else None
    column_mapping_cache[cache_key] = result
    return result


def execute_basic_query(df: pd.DataFrame, question: str) -> Optional[str]:
    """Execute computational queries"""
    agg = detect_aggregation(question)
    
    if not agg:
        return None

    try:
        # Handle GROUP BY
        if " by " in question.lower():
            parts = re.split(r'\s+by\s+', question.lower())
            
            if len(parts) >= 2:
                target_col = detect_column_fuzzy(df, parts[0])
                group_col = detect_column_fuzzy(df, parts[1])

                if target_col and group_col:
                    result = df.groupby(group_col)[target_col].agg(agg)
                    output = f"{agg.upper()} of {target_col} by {group_col}:\n{result.to_string()}"
                    return output

        # Simple aggregation
        column = detect_column_fuzzy(df, question)
        
        if column:
            if agg == "count":
                result = df[column].count()
            elif agg == "sum":
                result = df[column].sum()
            elif agg == "mean":
                result = df[column].mean()
            elif agg == "max":
                result = df[column].max()
            elif agg == "min":
                result = df[column].min()
            else:
                return None

            return f"{agg.upper()} of '{column}': {result}"

    except Exception as e:
        return None

    return None


# =====================================================
# PROMPT ENGINEERING - SHORT & CRISP
# =====================================================

def get_system_prompt(query_type: str) -> str:
    
    if query_type == 'simple_greeting':
        return """You are Nova, a friendly AI data analyst assistant. Respond to greetings warmly but briefly (1-2 sentences max)."""
    
    elif query_type == 'small_talk':
        return """You are Nova, a conversational AI assistant.
Rules:
- Keep responses SHORT (2-3 sentences max)
- Be friendly but professional
- You can engage in small talk briefly
- Don't write long paragraphs"""
    
    elif query_type == 'computation':
        return """You are Nova, an AI Data Analyst.
Rules:
- Give DIRECT numerical answers
- Use format: "Result: [number]"
- Add ONE sentence of context only if necessary
- Be extremely concise"""
    
    elif query_type == 'data_query':
        return """You are Nova, an AI Data Analyst.
Rules:
- If dataset context is provided, prioritize answering from it
- If not in the context, use your general knowledge to help
- Answer in 2-3 sentences MAX
- Be direct and specific"""
    
    else:
        return """You are Nova, a helpful AI assistant with data analysis expertise.
Rules:
- Answer any general questions helpfully and accurately
- For dataset-specific questions, use the provided context
- Keep responses SHORT (2-3 sentences)
- Be friendly and conversational"""


def create_user_prompt(question: str, context: str, query_type: str) -> str:
    
    if query_type == 'simple_greeting':
        return f"User said: {question}\n\nRespond warmly in 1-2 sentences."
    
    elif query_type == 'small_talk':
        return f"User said: {question}\n\nRespond naturally and briefly in 2-3 sentences."
    
    elif context and len(context.strip()) > 0:
        return f"""You have access to the user's dataset report below AND your general knowledge.
Prioritize the dataset context for data-specific questions.
For general questions, answer from your knowledge.

Dataset Context:
{context[:500]}

Question: {question}

Answer in 2-3 sentences. Be direct and helpful."""
    
    else:
        return f"""Question: {question}

Answer this helpfully using your general knowledge.
Keep it SHORT — 2-3 sentences max. Be direct and friendly."""

# =====================================================
# MAIN RESPONSE FUNCTION
# =====================================================

def dataset_chat_response(report_path: str, csv_path: str, user_question: str, session_id: str = "default") -> str:
    """
    Main chatbot response - SHORT & CRISP
    """
    if not user_question or not user_question.strip():
        return "What would you like to know? 🤔"

    # Initialize session context
    if session_id not in conversation_context:
        conversation_context[session_id] = {'greeted': False, 'query_count': 0}
    
    context_data = conversation_context[session_id]
    context_data['query_count'] += 1
    
    # Classify query
    query_type = classify_query_type(user_question)
    
    # Handle simple greetings (only greet ONCE per session)
    if query_type == 'simple_greeting':
        if not context_data['greeted']:
            context_data['greeted'] = True
            return "Hi! I'm Nova, your data analyst. Ask me anything about your dataset! 📊"
        else:
            return "Hey! What can I help you with? 😊"
    
    # Handle small talk briefly
    if query_type == 'small_talk':
        return "I appreciate you sharing! 😊 While I'm here primarily for data analysis, feel free to ask me about your dataset - I'd love to help with that!"

    # Load DataFrame
    df = None
    if csv_path and isinstance(csv_path, str):
        if csv_path not in dataframe_cache:
            if os.path.exists(csv_path):
                try:
                    dataframe_cache[csv_path] = pd.read_csv(csv_path)
                except:
                    dataframe_cache[csv_path] = None
        df = dataframe_cache.get(csv_path)

    # Try computation first (direct answers)
    if df is not None and query_type == 'computation':
        computation_result = execute_basic_query(df, user_question)
        if computation_result:
            return computation_result

    # RAG for data queries
    context = ""
    
    if report_path and os.path.exists(report_path):
        if report_path not in vectorstore_cache:
            vectorstore = build_vectorstore_from_html(report_path)
            if vectorstore:
                vectorstore_cache[report_path] = vectorstore
        
        vectorstore = vectorstore_cache.get(report_path)
        
        if vectorstore:
            try:
                docs = vectorstore.similarity_search(user_question, k=2)
                context = "\n".join([doc.page_content for doc in docs])
            except:
                pass

    # Generate LLM response
    try:
        llm = get_llm()
        
        response = llm.invoke([
            SystemMessage(content=get_system_prompt(query_type)),
            HumanMessage(content=create_user_prompt(user_question, context, query_type))
        ])

        return response.content

    except Exception as e:
        print(f"❌ Error: {e}")
        return "Sorry, I encountered an error. Please try again!"


# =====================================================
# STREAMING VERSION FOR ANIMATION EFFECT
# =====================================================

def dataset_chat_response_stream(report_path: str, csv_path: str, user_question: str, session_id: str = "default"):
    """
    Streaming version for character-by-character animation
    Use this in your Flask app for streaming responses
    """
    if not user_question or not user_question.strip():
        yield "What would you like to know? 🤔"
        return

    # Same logic as above but with streaming
    if session_id not in conversation_context:
        conversation_context[session_id] = {'greeted': False, 'query_count': 0}
    
    context_data = conversation_context[session_id]
    context_data['query_count'] += 1
    
    query_type = classify_query_type(user_question)
    
    # Quick responses (no streaming needed)
    if query_type == 'simple_greeting':
        if not context_data['greeted']:
            context_data['greeted'] = True
            yield "Hi! I'm Nova, your data analyst. Ask me anything about your dataset! 📊"
        else:
            yield "Hey! What can I help you with? 😊"
        return
    
    if query_type == 'small_talk':
        yield "I appreciate you sharing! 😊 While I'm here primarily for data analysis, feel free to ask me about your dataset - I'd love to help with that!"
        return

    # Load DataFrame
    df = None
    if csv_path and isinstance(csv_path, str):
        if csv_path not in dataframe_cache:
            if os.path.exists(csv_path):
                try:
                    dataframe_cache[csv_path] = pd.read_csv(csv_path)
                except:
                    pass
        df = dataframe_cache.get(csv_path)

    # Computation
    if df is not None and query_type == 'computation':
        result = execute_basic_query(df, user_question)
        if result:
            yield result
            return

    # RAG
    context = ""
    if report_path and os.path.exists(report_path):
        if report_path not in vectorstore_cache:
            vectorstore = build_vectorstore_from_html(report_path)
            if vectorstore:
                vectorstore_cache[report_path] = vectorstore
        
        vectorstore = vectorstore_cache.get(report_path)
        if vectorstore:
            try:
                docs = vectorstore.similarity_search(user_question, k=2)
                context = "\n".join([doc.page_content for doc in docs])
            except:
                pass

    # Stream LLM response
    try:
        llm = get_llm()
        
        for chunk in llm.stream([
            SystemMessage(content=get_system_prompt(query_type)),
            HumanMessage(content=create_user_prompt(user_question, context, query_type))
        ]):
            yield chunk.content

    except Exception as e:
        yield "Sorry, I encountered an error. Please try again!"


# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def clear_cache():
    """Clear all caches"""
    global vectorstore_cache, dataframe_cache, column_mapping_cache, conversation_context
    vectorstore_cache.clear()
    dataframe_cache.clear()
    column_mapping_cache.clear()
    conversation_context.clear()


def reset_session(session_id: str):
    """Reset a specific session"""
    if session_id in conversation_context:
        del conversation_context[session_id]