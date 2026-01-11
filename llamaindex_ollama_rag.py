import json
import logging
from logging.handlers import RotatingFileHandler
import os
import platform
import subprocess
import sys
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import re

import chromadb
import requests
from contextlib import redirect_stdout, contextmanager
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
    SQLDatabase,
    SimpleDirectoryReader,
)
from llama_index.core.chat_engine.simple import SimpleChatEngine
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url

import config
import database_utils
import utils
from kaia_cli import KaiaCLI
from toolbox import video_converter


# --- Basic Setup ---
# Configure logging with rotation
log_file = Path("kaia.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3),
        logging.StreamHandler(sys.stdout)
    ]
)

# Silence noisy loggers
for noisy in ["httpx", "httpcore", "fsspec", "urllib3", "llama_index.core.storage.kvstore", "chromadb", "llama_index"]:
    logging.getLogger(noisy).setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

@contextmanager
def suppress_stdout():
    """Temporarily redirects stdout to devnull to suppress noisy output."""
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout # Store original stdout
        sys.stdout = fnull # Redirect stdout
        try:
            yield
        finally:
            sys.stdout = old_stdout # Restore original stdout

# --- Global State ---
class AppState:
    def __init__(self):
        self.sql_rag_enabled = config.SQL_RAG_ENABLED
        self.cli = KaiaCLI()
        self.kaia_persona_content = config.KAIA_SYSTEM_PROMPT
        self.index = None
        self.rag_chat_engine = None
        self.pure_chat_engine = None
        self.sql_query_engine = None
        self.current_working_directory = Path.cwd()
        self.user_id = database_utils.get_current_user()
        self.vector_store = None

# --- Initialization Functions ---
def initialize_models():
    """Initialize LLM and Embedding Models."""
    print(f"{config.COLOR_BLUE}Initializing LLM and Embedding Model...{config.COLOR_RESET}")
    start_time = time.time()
    try:
        llm_model, llm_err = utils.check_ollama_model_availability(config.LLM_MODEL, config.DEFAULT_COMMAND_MODEL)
        if llm_err: raise RuntimeError(f"LLM init failed: {llm_err}")

        embed_model, embed_err = utils.check_ollama_model_availability(config.EMBEDDING_MODEL)
        if embed_err: raise RuntimeError(f"Embedding model init failed: {embed_err}")

        with suppress_stdout():
            Settings.llm = Ollama(model=llm_model, request_timeout=config.TIMEOUT_SECONDS, stream=True)
            Settings.embed_model = OllamaEmbedding(model_name=embed_model)
            _ = Settings.llm.complete("Hello")
            embedding_dim = len(Settings.embed_model.get_query_embedding("test"))

        logger.info(f"LLM ({llm_model}) and embedding model ({embed_model}) initialized.")
        print(f"{config.COLOR_GREEN}Models initialized in {time.time() - start_time:.2f}s. Embedding dim: {embedding_dim}{config.COLOR_RESET}")
        return embedding_dim
    except Exception as e:
        logger.error(f"Failed to initialize Ollama models: {e}", exc_info=True)
        print(f"{config.COLOR_RED}Fatal: Failed to initialize Ollama models. Exiting.{config.COLOR_RESET}")
        sys.exit(1)

def load_persona(state: AppState):
    """Load Kaia's persona from a file or use the default."""
    print(f"{config.COLOR_BLUE}Loading Kaia persona...{config.COLOR_RESET}")
    persona_path = config.PERSONA_DIR / "Kaia_Desktop_Persona.md"
    try:
        if persona_path.exists():
            state.kaia_persona_content = persona_path.read_text().replace('\x00', '')
            print(f"{config.COLOR_GREEN}Persona loaded successfully.{config.COLOR_RESET}")
        else:
            print(f"{config.COLOR_YELLOW}Warning: Persona file not found. Using default prompt.{config.COLOR_RESET}")
    except Exception as e:
        print(f"{config.COLOR_YELLOW}Warning: Could not load persona: {e}. Using default.{config.COLOR_RESET}")

def initialize_vector_db(embedding_dim: int):
    """Initialize ChromaDB, recreating if necessary for dimension consistency."""
    print(f"{config.COLOR_BLUE}Initializing ChromaDB...{config.COLOR_RESET}")
    try:
        chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        collection_name = "kaia_documents"

        # Use get_or_create_collection for safer handling
        collection = chroma_client.get_or_create_collection(collection_name)

        # Check for dimension mismatch only if the collection is not empty
        if collection.count() > 0:
            peeked_items = collection.peek(limit=1)
            # Ensure there are embeddings to check
            embeddings_data = peeked_items.get("embeddings")
            if peeked_items and embeddings_data is not None and len(embeddings_data) > 0:
                existing_dim = len(peeked_items["embeddings"][0])
                if existing_dim != embedding_dim:
                    print(f"{config.COLOR_YELLOW}Warning: ChromaDB dimension mismatch. "
                          f"Expected {embedding_dim}, found {existing_dim}. Recreating collection.{config.COLOR_RESET}")
                    # Delete and recreate the collection with the correct dimension
                    chroma_client.delete_collection(name=collection_name)
                    collection = chroma_client.create_collection(name=collection_name)

        vector_store = ChromaVectorStore(chroma_collection=collection)
        print(f"{config.COLOR_GREEN}ChromaDB initialized successfully.{config.COLOR_RESET}")
        return vector_store

    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}", exc_info=True)
        print(f"{config.COLOR_RED}Fatal: Failed to initialize ChromaDB. Exiting.{config.COLOR_RESET}")
        sys.exit(1)

def build_or_load_index(vector_store: ChromaVectorStore, force_rebuild: bool = False):
    """Build or load the LlamaIndex from storage, checking for staleness."""
    print(f"{config.COLOR_BLUE}Loading/Building LlamaIndex...{config.COLOR_RESET}")
    try:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index_exists = os.path.exists(config.LLAMA_INDEX_METADATA_PATH)
        
        should_build = force_rebuild or not index_exists
        
        if index_exists and not force_rebuild:
            # Check for staleness
            try:
                # Get latest modification time of any file in data directories
                last_doc_mtime = 0
                for doc_dir in [config.GENERAL_KNOWLEDGE_DIR, config.PERSONAL_CONTEXT_DIR]:
                    if doc_dir.exists():
                        for root, _, files in os.walk(doc_dir):
                            for f in files:
                                mtime = os.path.getmtime(os.path.join(root, f))
                                if mtime > last_doc_mtime:
                                    last_doc_mtime = mtime
                
                # Get index creation/modification time
                index_mtime = os.path.getmtime(config.LLAMA_INDEX_METADATA_PATH)
                
                if last_doc_mtime > index_mtime:
                    print(f"{config.COLOR_YELLOW}Index is stale (new documents found). Rebuilding...{config.COLOR_RESET}")
                    should_build = True
            except Exception as e:
                logger.warning(f"Could not check index freshness: {e}")

        if should_build:
             print(f"{config.COLOR_BLUE}{'Rebuilding' if index_exists else 'Building'} index from documents...{config.COLOR_RESET}")
             all_docs = []
             for doc_dir in [config.GENERAL_KNOWLEDGE_DIR, config.PERSONAL_CONTEXT_DIR]:
                 if doc_dir.exists():
                     print(f"Scanning {doc_dir}...")
                     reader = SimpleDirectoryReader(input_dir=str(doc_dir), recursive=True)
                     all_docs.extend(reader.load_data())
             
             if not all_docs:
                 print(f"{config.COLOR_YELLOW}Warning: No documents found to build index.{config.COLOR_RESET}")
                 return None
                 
             with suppress_stdout():
                # If rebuilding, we might want to clear the vector store to avoid duplicates if not handled by the store
                # ChromaVectorStore with LlamaIndex usually handles updates but a full rebuild is safer for consistency here
                index = VectorStoreIndex.from_documents(all_docs, storage_context=storage_context)
                index.storage_context.persist(persist_dir=config.LLAMA_INDEX_METADATA_PATH)
             print(f"{config.COLOR_GREEN}Index built and saved.{config.COLOR_RESET}")
        else:
            print(f"{config.COLOR_BLUE}Loading existing index...{config.COLOR_RESET}")
            with suppress_stdout():
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store, persist_dir=config.LLAMA_INDEX_METADATA_PATH
                )
                index = load_index_from_storage(storage_context)
            print(f"{config.COLOR_GREEN}Index loaded.{config.COLOR_RESET}")
        return index
    except Exception as e:
        logger.error(f"Failed to initialize LlamaIndex: {e}", exc_info=True)
        print(f"{config.COLOR_RED}Error: Failed to initialize LlamaIndex.{config.COLOR_RESET}")
        return None

def initialize_sql_engine(state: AppState):
    """Initialize the SQL database and query engine."""
    if not state.sql_rag_enabled: return
    print(f"{config.COLOR_BLUE}Initializing PostgreSQL database...{config.COLOR_RESET}")
    try:
        with suppress_stdout():
            database_utils.initialize_db()
            sql_database = SQLDatabase(database_utils.engine)
            state.sql_query_engine = NLSQLTableQueryEngine(
                sql_database=sql_database,
                llm=Settings.llm,
                tables=["facts", "interaction_history", "user_preferences"]
            )
        print(f"{config.COLOR_GREEN}PostgreSQL initialized.{config.COLOR_RESET}")
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL: {e}", exc_info=True)
        print(f"{config.COLOR_RED}Warning: Failed to initialize PostgreSQL.{config.COLOR_RESET}")
        state.sql_rag_enabled = False

def setup_chat_engines(state: AppState):
    """Create and configure the RAG and pure chat engines."""
    # Pure Chat Engine (always available)
    state.pure_chat_engine = SimpleChatEngine.from_defaults(
        llm=Settings.llm,
        chat_mode="best",
        memory=ChatMemoryBuffer.from_defaults(token_limit=8192),
        system_prompt=config.KAIA_SYSTEM_PROMPT + "\n\n" + state.kaia_persona_content,
    )

    # RAG Chat Engine (if index is available)
    if state.index:
        combined_system_prompt = (
            "You are Kaia, a helpful AI assistant. Use the provided context to answer questions. "
            "If the answer is not in the context, state that clearly. Do not invent information. "
            "Maintain your core persona: strategic, precise, and intellectual. "
            "Persona details: " + state.kaia_persona_content
        )
        state.rag_chat_engine = state.index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=ChatMemoryBuffer.from_defaults(token_limit=8192),
            system_prompt=combined_system_prompt,
            similarity_top_k=4
        )
    else:
        print(f"{config.COLOR_YELLOW}Warning: RAG index not available. RAG queries will fallback to pure chat.{config.COLOR_RESET}")
        state.rag_chat_engine = state.pure_chat_engine

# --- Action Handlers (Refactored Logic) ---
def handle_store_data(state: AppState, content: str) -> str:
    _, response = database_utils.handle_memory_storage(state.user_id, content)
    print(f"\n{config.COLOR_BLUE}Kaia: {response}{config.COLOR_RESET}")
    return response

def handle_command(state: AppState, content: str) -> str:
    print(f"\n{config.COLOR_BLUE}Kaia (Command Mode):{config.COLOR_RESET}")
    command, error = state.cli.generate_command(str(content))
    if error:
        response = f"Command generation failed: {error}"
        print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
        return response

    print(f"\n{config.COLOR_YELLOW}┌── Proposed Command ──┐{config.COLOR_RESET}")
    print(f"{config.COLOR_BLUE}{command}{config.COLOR_RESET}")
    print(f"{config.COLOR_YELLOW}└──────────────────────┘{config.COLOR_RESET}")
    confirm = input(f"{config.COLOR_YELLOW}Execute? (y/N): {config.COLOR_RESET}").lower().strip()

    if confirm != 'y':
        response = f"Command cancelled: {command}"
        print(f"{config.COLOR_BLUE}{response}{config.COLOR_RESET}")
        return response

    if command.strip().startswith("cd "):
        target_dir = command.strip()[3:].strip()
        try:
            new_path = (state.current_working_directory / os.path.expanduser(target_dir)).resolve()
            if new_path.is_dir():
                state.current_working_directory = new_path
                response = f"Changed directory to: {state.current_working_directory}"
                print(f"{config.COLOR_GREEN}{response}{config.COLOR_RESET}")
            else:
                response = f"Error: Directory not found: {target_dir}"
                print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
        except Exception as e:
            response = f"Error changing directory: {e}"
            print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
    else:
        success, stdout, stderr = state.cli.execute_command(command, cwd=str(state.current_working_directory))
        if success:
            response = f"Command executed successfully.\n{stdout}"
            print(f"{config.COLOR_GREEN}{response}{config.COLOR_RESET}")
            if stderr: print(f"{config.COLOR_YELLOW}Stderr:\n{stderr}{config.COLOR_RESET}")
        else:
            response = f"Command failed.\nStderr: {stderr}\nStdout: {stdout}"
            print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
    return response

def handle_run_script(state: AppState, content: str) -> str:
    script_name = content
    script_path = os.path.expanduser(os.path.join("~", script_name))

    if script_name not in config.SCRIPT_ALLOWLIST:
        response = f"Error: Script '{script_name}' is not in the allowlist."
        print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
        return response

    if not (os.path.exists(script_path) and os.path.isfile(script_path) and os.access(script_path, os.X_OK)):
        response = f"Error: Script '{script_name}' not found, not a file, or not executable."
        print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
        return response

    print(f"\n{config.COLOR_BLUE}Kaia (Running Script): {script_path}{config.COLOR_RESET}")
    confirm = input(f"{config.COLOR_YELLOW}Execute script? (y/N): {config.COLOR_RESET}").lower().strip()
    if confirm != 'y':
        response = f"Script execution cancelled: {script_name}"
        print(f"{config.COLOR_BLUE}{response}{config.COLOR_RESET}")
        return response

    success, stdout, stderr = state.cli.execute_command(script_path)
    if success:
        response = f"Script executed successfully.\n{stdout}"
        print(f"{config.COLOR_GREEN}{response}{config.COLOR_RESET}")
        if stderr: print(f"{config.COLOR_YELLOW}Stderr:\n{stderr}{config.COLOR_RESET}")
    else:
        response = f"Script failed.\nStderr: {stderr}\nStdout: {stdout}"
        print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
    return response

def handle_system_status(state: AppState, content: str) -> str:
    status_info = state.cli.get_system_status()
    status_info['db_status'] = database_utils.get_database_status()
    response = state.cli.format_system_status_output(status_info)
    print(f"\n{config.COLOR_BLUE}Kaia:{config.COLOR_RESET}")
    print(f"{config.COLOR_GREEN}┌── System Status ──┐{config.COLOR_RESET}")
    print(response)
    print(f"{config.COLOR_GREEN}└───────────────────┘{config.COLOR_RESET}")
    return response

def handle_sql_query(state: AppState, content: str) -> str:
    if not state.sql_rag_enabled or not state.sql_query_engine:
        response = "Database query functionality is not available."
        print(f"{config.COLOR_YELLOW}{response}{config.COLOR_RESET}")
        return response

    try:
        print(f"\n{config.COLOR_BLUE}Kaia (Querying Database):{config.COLOR_RESET}")
        sql_response = state.sql_query_engine.query(content)
        response = str(sql_response)
        print(f"{config.COLOR_GREEN}┌── Query Results ──┐{config.COLOR_RESET}")
        print(response)
        print(f"{config.COLOR_GREEN}└───────────────────┘{config.COLOR_RESET}")
    except Exception as e:
        response = f"Database Error: {e}"
        print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
    return response

def handle_retrieve_data(state: AppState, content: str) -> str:
    result = database_utils.handle_data_retrieval(state.user_id, content)
    response = result['message']
    print(f"\n{config.COLOR_BLUE}Kaia:{config.COLOR_RESET}")
    if isinstance(result.get('data'), list) and result['data']:
        print(f"{config.COLOR_GREEN}┌── {result['message']} ──┐{config.COLOR_RESET}")
        for item in result['data']:
            print(f"• {str(item)}")
        print(f"{config.COLOR_GREEN}└──────────────────────┘{config.COLOR_RESET}")
    else:
        print(response)
    return "\n".join(result.get('data', [response]))

def handle_knowledge_query(state: AppState, content: str, start_time: float) -> str:
    print(f"\n{config.COLOR_BLUE}Kaia:{config.COLOR_RESET}", end=" ", flush=True)
    try:
        # Check if the query is a request for a summary of a specific document
        summary_keywords = ["summary of", "summarize", "synopsis of", "tell me about", "overview of"]
        is_summary_request = False
        document_title = None

        for keyword in summary_keywords:
            if keyword in content.lower():
                is_summary_request = True
                # Attempt to extract the document title more robustly
                # Look for text within single or double quotes first
                match = re.search(r"['\"]([^'\"]+)['\"]", content)
                if match:
                    document_title = match.group(1).strip()
                else:
                    # Fallback to extracting text after the keyword
                    parts = content.lower().split(keyword, 1)
                    if len(parts) > 1:
                        document_title = parts[1].strip()
                        # Remove "from X.pdf" if present
                        document_title = re.sub(r"\s*from\s+['\"].*\.pdf['\"]", "", document_title)
                        document_title = re.sub(r"\s*from\s+.*\.pdf", "", document_title) # also without quotes
                        document_title = document_title.replace(".pdf", "").replace(".md", "").strip("'\"")

                if document_title:
                    # Ensure the extracted title is clean and meaningful
                    document_title = document_title.replace("_", " ").strip()
                    if not document_title: # If after cleaning, it's empty, reset
                        is_summary_request = False
                        document_title = None
                else:
                    is_summary_request = False
                break

        if is_summary_request and document_title:
            # New conditional logic for collection-based summarization
            if "collect" in document_title.lower() or "compendium" in document_title.lower() or "book" in document_title.lower():
                rag_query = (
                    f"You are Kaia, a precise and grounded AI assistant. Your task is to analyze and summarize the document titled '{document_title}', based solely on its content.\n\n"
                    "This document may not contain a linear story, but instead consist of multiple entries, such as books, journal excerpts, lore entries, or letters. If so:\n"
                    "1. Clearly describe the structure and type of content included (e.g., short stories, in-universe texts, categorized entries).\n"
                    "2. Identify and summarize any recurring themes, formats, or subjects that emerge from the entries.\n"
                    "3. Highlight notable patterns: e.g., common topics (war, gods, history), stylistic tone (mythic, satirical), or presentation format (volumes, regions, authors).\n"
                    "4. Do **not** invent overarching narratives unless clearly presented.\n"
                    "5. Do **not** draw from prior knowledge—your answer must reflect *only* what is in the document.\n\n"
                    "If the content is too fragmented or unstructured for meaningful summarization, state that clearly."
                    "Your goal is to help a user understand what kind of document this is, what they will find in it, and how it is organized—without speculation."
                )
            else:
                # Original narrative-focused summarization prompt
                rag_query = (
                    f"Based *only* on the provided context, which is from the document titled '{document_title}', "
                    f"provide a comprehensive summary. Focus on the main story, key characters, significant events, "
                    f"and the overall purpose or theme of the document *as described in the provided text*. "
                    f"It is critical that you *do not* include any information not explicitly present in the provided text, "
                    f"even if you have prior knowledge. "
                    f"If the provided text is too short or lacks sufficient detail for a comprehensive summary, "
                    f"please state that clearly and explain why, without inventing any details."
                )
            logging.info(f"Refined RAG query for summarization: {rag_query}")
        else:
            rag_query = content # Use original content if not a specific summary request

        response_stream = state.rag_chat_engine.stream_chat(rag_query)
        return stream_and_print_response(response_stream, start_time)
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        response = "Error retrieving information from my knowledge base."
        print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
        return response

def handle_rebuild_index(state: AppState, content: str) -> str:
    print(f"\n{config.COLOR_BLUE}Kaia (Rebuilding Index):{config.COLOR_RESET}")
    try:
        # Re-initialize vector store to ensure clean state if needed, or just pass existing
        # For simplicity, we'll reuse the existing logic but force rebuild
        # Note: We need the vector store instance. It's not stored in state explicitly in the original code,
        # but we can re-initialize it or store it in AppState.
        # Let's modify AppState to store vector_store.
        
        if not hasattr(state, 'vector_store'):
             # Fallback if we didn't update AppState yet (which we should do next)
             embedding_dim = initialize_models() # This might be redundant but safe
             state.vector_store = initialize_vector_db(embedding_dim)

        state.index = build_or_load_index(state.vector_store, force_rebuild=True)
        setup_chat_engines(state) # Re-setup engines with new index
        
        response = "Knowledge base index has been successfully rebuilt."
        print(f"{config.COLOR_GREEN}{response}{config.COLOR_RESET}")
        return response
    except Exception as e:
        response = f"Failed to rebuild index: {e}"
        print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
        return response

def handle_chat(state: AppState, content: str, start_time: float) -> str:
    print(f"\n{config.COLOR_BLUE}Kaia:{config.COLOR_RESET}", end=" ", flush=True)
    response_stream = state.pure_chat_engine.stream_chat(content)
    return stream_and_print_response(response_stream, start_time)

# --- Core Utilities ---
def generate_action_plan(user_input: str) -> Dict[str, str]:
    """Generates an action plan (e.g., command, knowledge query) based on user input."""
    # Define a very strict system prompt for JSON output
    # Simplified and made more direct to minimize JSON errors
    system_prompt_for_action_plan = (
        "You are an AI that determines the user's intent and provides a JSON response. "
        "Your response MUST be a JSON object with two keys: 'action' and 'content'. "
        "The 'action' must be one of: 'knowledge_query', 'chat', 'command', 'store_data', 'retrieve_data', 'sql', 'run_script', 'system_status', 'convert_video_to_gif', 'get_persona_content', 'text_extraction', 'rebuild_index'. "
        "The 'content' should be the relevant text for the chosen action. "
        "Respond ONLY with the JSON object. Do NOT include any other text or explanations."
        "Example: {'action': 'knowledge_query', 'content': 'What is the capital of France?'}"
    )

    payload = {
        "model": config.DEFAULT_COMMAND_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt_for_action_plan},
        ] + config.ACTION_PLAN_EXAMPLES + [
            {"role": "user", "content": str(user_input)}
        ],
        "stream": False,
        "format": "json"
    }
    try:
        model_to_use, error_msg = utils.check_ollama_model_availability(config.DEFAULT_COMMAND_MODEL, config.LLM_MODEL)
        if error_msg: raise RuntimeError(error_msg)
        payload["model"] = model_to_use

        response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=config.TIMEOUT_SECONDS)
        response.raise_for_status()
        return json.loads(response.json()["message"]["content"].strip())
    except (json.JSONDecodeError, requests.RequestException, RuntimeError) as e:
        logger.error(f"Action plan generation failed, falling back to chat: {e}", exc_info=True)
        return {"action": "chat", "content": user_input}

def stream_and_print_response(response_stream, start_time: float) -> str:
    """Streams tokens from the LLM response and prints them with proper word wrapping and timing."""
    full_response = ""
    first_token_time = None
    
    # Get dynamic terminal width
    try:
        term_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    except Exception:
        term_width = 80
        
    # Buffer for word assembly
    buffer = ""
    current_col = 0

    print(f"{config.COLOR_BLUE}", end="", flush=True)

    for token in response_stream.response_gen:
        if not first_token_time:
            first_token_time = time.time()
        full_response += token
        buffer += token
        
        while True:
            # Prioritize explicit newlines
            newline_pos = buffer.find('\n')
            space_pos = buffer.find(' ')
            
            # Handle Newlines
            if newline_pos != -1 and (space_pos == -1 or newline_pos < space_pos):
                chunk = buffer[:newline_pos]
                
                if current_col > 0:
                    if current_col + len(chunk) <= term_width:
                         sys.stdout.write(chunk)
                    else:
                         sys.stdout.write(chunk)
                else:
                    sys.stdout.write(chunk)

                sys.stdout.write('\n')
                sys.stdout.flush() # Flush immediately on newline
                current_col = 0
                buffer = buffer[newline_pos+1:]
                continue
            
            # Handle Spaces (Word boundaries)
            if space_pos != -1:
                word = buffer[:space_pos]
                word_len = len(word)
                
                if current_col == 0:
                    sys.stdout.write(word)
                    current_col += word_len
                else:
                    if current_col + 1 + word_len > term_width:
                        sys.stdout.write('\n')
                        sys.stdout.write(word)
                        current_col = word_len
                    else:
                        sys.stdout.write(' ' + word)
                        current_col += 1 + word_len
                
                sys.stdout.flush() # Flush immediately after a word
                buffer = buffer[space_pos+1:]
                continue
            
            # No delimiter found, check if buffer is getting too long (safety valve)
            # If buffer is longer than width, we must print it to avoid holding too much
            if len(buffer) > term_width:
                 sys.stdout.write(buffer)
                 sys.stdout.flush()
                 current_col += len(buffer)
                 buffer = ""
            
            break
            
    # Flush remaining buffer
    if buffer:
        if current_col == 0:
             sys.stdout.write(buffer)
        else:
             if current_col + 1 + len(buffer) > term_width:
                 sys.stdout.write('\n')
                 sys.stdout.write(buffer)
             else:
                 sys.stdout.write(' ' + buffer)
        sys.stdout.flush()
    
    print(f"\n\n{config.COLOR_YELLOW}⏱ Total time: {time.time() - start_time:.2f}s", end="")
    if first_token_time:
        print(f" (First token: {first_token_time - start_time:.2f}s)", end="")
    print(f"{config.COLOR_RESET}")

    return full_response.strip()

# --- Main Application Logic ---
def process_user_input(state: AppState, query: str, action_handlers: Dict[str, Callable]):
    """Processes a single user input query."""
    start_time = time.time()
    response = ""
    response_type = "unclassified"

    try:
        # Fast path for exit commands
        lower_query = query.lower().strip()
        if lower_query in ['exit', 'quit', '/exit', '/quit']:
            plan = {"action": "chat", "content": "Goodbye!"}
        else:
            plan = generate_action_plan(query)
        
        action = plan.get("action", "chat")
        content = plan.get("content", query)
        response_type = action

        handler = action_handlers.get(action)
        if handler:
            # Pass start_time only to handlers that need it
            if action in ["knowledge_query", "chat", "text_extraction"]:
                response = handler(state, content, start_time)
            else:
                response = handler(state, content)
        else:
            logger.warning(f"No handler for action '{action}'. Defaulting to chat.")
            response = handle_chat(state, query, start_time)
            response_type = "chat"

        # Log the interaction
        database_utils.log_interaction(
            user_id=state.user_id,
            user_query=query,
            kaia_response=response,
            response_type=response_type
        )

    except Exception as e:
        logger.exception("Unexpected error processing input")
        response = f"System error: {e}"
        print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
        database_utils.log_interaction(
            user_id=state.user_id,
            user_query=query,
            kaia_response=f"System error: {str(e)[:200]}",
            response_type="system_error"
        )

def main():
    # 1. Initialize application state and components
    state = AppState()
    embedding_dim = initialize_models()
    load_persona(state)
    state.vector_store = initialize_vector_db(embedding_dim)
    state.index = build_or_load_index(state.vector_store)
    initialize_sql_engine(state)
    setup_chat_engines(state)
    database_utils.ensure_user(state.user_id)

    # 2. Define action dispatcher
    action_handlers: Dict[str, Callable] = {
        "store_data": handle_store_data,
        "command": handle_command,
        "run_script": handle_run_script,
        "system_status": handle_system_status,
        "sql": handle_sql_query,
        "retrieve_data": handle_retrieve_data,
        "knowledge_query": lambda s, c, t: handle_knowledge_query(s, c, t),
        "text_extraction": lambda s, c, t: handle_knowledge_query(s, c, t),
        "text_generation": lambda s, c, t: handle_chat(s, c, t), # Map text_generation to chat
        "chat": lambda s, c, t: handle_chat(s, c, t),
        "convert_video_to_gif": lambda s, c: video_converter.convert_video_to_gif_interactive(s.cli, s.user_id)['response'],
        "get_persona_content": lambda s, c: state.kaia_persona_content,
        "rebuild_index": handle_rebuild_index,
    }

    # 3. Print welcome message
    print(f"""{config.COLOR_BLUE}
██╗  ██╗ █████╗ ██╗ █████╗
██║ ██╔╝██╔══██╗██║██╔══██╗
█████╔╝ ███████║██║███████║
██╔═██╗ ██╔══██║██║██╔══██║
██║  ██╗██║  ██║██║██║  ██║
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
{config.COLOR_RESET}
""")

    # 4. Main loop
    while True:
        try:
            query = input("\nYou: ").strip()
            if not query: continue
            if query.lower() in ['exit', 'quit', '/exit', '/quit']:
                print(f"{config.COLOR_BLUE}Kaia: Session ended.{config.COLOR_RESET}")
                break

            process_user_input(state, query, action_handlers)

        except KeyboardInterrupt:
            print(f"\n{config.COLOR_BLUE}Kaia: Exiting gracefully...{config.COLOR_RESET}")
            break
        except Exception as e:
            logger.exception("Unexpected error in main loop")
            print(f"{config.COLOR_RED}System error: {e}{config.COLOR_RESET}")

if __name__ == "__main__":
    main()
