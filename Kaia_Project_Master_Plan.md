Kaia AI Assistant: Project Master Plan
Version: 1.2
Last Updated: August 1, 2025

This document serves as the single source of truth for the Kaia AI Assistant project, consolidating its vision, architecture, features, progress, and future roadmap.

Table of Contents

Project Vision & Core Goal

Core Architecture & Technology Stack

Refactoring and Code Improvements (V1.2)

Key Features & Implementation

Development Progress Log

Future Roadmap & Planned Features

Setup & Configuration Summary

1. Project Vision & Core Goal
The core goal is to build Kaia, a local, intelligent AI assistant designed to enhance the Linux desktop experience. Kaia is envisioned to be a powerful conversational partner capable of leveraging a personal knowledge base (RAG) and executing Linux commands to perform tasks, answer questions, and streamline workflows. The project prioritizes local execution, user control, and a distinct persona.

2. Core Architecture & Technology Stack
Kaia's foundation is built on a stack of local, open-source tools that ensure privacy and control.

2.1. LLM and Embedding Engine: Ollama

Primary LLM (Chat & RAG): llama2:13b-chat-q4_0

Command Generation LLM: mixtral:8x7b-instruct-v0.1-q4_K_M (A more powerful model chosen for accurate command generation).

Embedding Model: nomic-embed-text for document chunking and vectorization.

Configuration: Models are initialized with a request_timeout=360.0 seconds and stream=True for real-time conversational output.

2.2. RAG and Indexing: LlamaIndex with ChromaDB

Function: LlamaIndex is used to build the Retrieval-Augmented Generation (RAG) system, allowing Kaia to access and reason about a personal knowledge base.

Vector Store: ChromaDB is integrated as the persistent vector store for efficient storage and retrieval of document embeddings.

Process:

Document Loading: SimpleDirectoryReader loads documents from specified knowledge directories.

Vector Indexing: VectorStoreIndex is built using the Ollama embedding model and stored in ChromaDB.

Persistent Storage: The index is persisted to and loaded from a ./storage directory using StorageContext and ChromaVectorStore to avoid re-indexing on every launch, significantly improving startup time.

2.3. Persistent Memory: PostgreSQL

Function: A PostgreSQL database is used for structured, long-term memory, enabling Kaia to remember user-specific facts, preferences, and interaction history.

Key Tables: users, user_preferences, facts, and interaction_history.

Operations: Utilizes SQLAlchemy for robust database interactions, including efficient upsert operations (ON CONFLICT) for user preferences.

Initialization: The database connection includes retry logic for enhanced resilience during startup.

2.4. Python Environment

The project is built in Python and relies on key libraries:

llama-index-core: For the core RAG pipeline.

llama-index-llms-ollama, llama-index-embeddings-ollama, llama-index-vector-stores-chroma: For specific LlamaIndex integrations.

SQLAlchemy, psycopg2: For PostgreSQL database interactions.

psutil, platform: For system monitoring.

subprocess: For executing shell commands.

requests: For API interactions.

3. Refactoring and Code Improvements (V1.2)
This section highlights the major architectural and functional improvements from the recent refactoring, based on the latest code reviews.

Modular Architecture: A clear separation of concerns has been implemented with dedicated modules and a centralized AppState class for state management.

Enhanced Security:

Script execution now requires an explicit allowlist and permission checks.

Command sanitization is stricter, with regex patterns and detection of unsafe operators like ;, &&, ||.

New Capabilities:

Script Runner: A new run_script action handler has been added for executing allowlisted scripts.

Video Conversion: A convert_video_to_gif action handler is now available.

Advanced Summarization: Special logic has been added for handling document collections and compendiums.

Directory Navigation: The system now persistently tracks the current working directory.

Improved Reliability:

ChromaDB initialization is more robust using get_or_create_collection() and includes dimension validation.

Comprehensive try/except blocks and fallback mechanisms have been added.

Performance Optimizations:

LRU caching is used for model checks.

Response streaming now includes first-token timing metrics.

Code Quality: pathlib is now used throughout the codebase for safer path handling, and OS-agnostic user identification has been implemented.

4. Key Features & Implementation
4.1. Dual-Mode Operation: Conversational Chat & Command Execution

Kaia intelligently routes user input into one of two primary modes:

Conversational Chat: For general queries, Kaia uses its RAG-powered chat engine to provide context-aware responses based on the knowledge base and its persona. Distinct RAG and pure chat engines are employed for optimized conversational flow.

Command Execution: For task-based requests, Kaia uses a dedicated model and a strict confirmation workflow to generate and execute Linux shell commands.

4.2. RAG System for Knowledge & Persona

Persona Loading: Kaia's distinct persona is loaded from a dedicated markdown file to ensure a consistent conversational style.

Knowledge Base: Documents from ./data and ./personal_context are loaded into a vector index (ChromaDB), providing Kaia with a long-term memory.

Chat Engine: The chat engine is configured with chat_mode="condense_plus_context" and a ChatMemoryBuffer (8192 token limit) to maintain conversational history.

4.3. Command Generation & Execution Framework

This is a critical safety and functionality component.

Dedicated Command Model: A more powerful model (mixtral) is used specifically for translating natural language into accurate shell commands.

Strict Prompting: A carefully crafted system prompt instructs the model to generate only the raw command, with no extra text.

Safety Confirmation: No command is executed without explicit user approval.

Execution & Output: The subprocess module is used to run the confirmed command, capturing and displaying stdout and stderr for user feedback.

4.4. Enhanced System Monitoring

Kaia now includes comprehensive system monitoring capabilities, providing real-time information on:

CPU usage and core count.

Memory usage.

Disk usage.

GPU utilization (NVIDIA/AMD) and Vulkan/OpenCL info.

Status of Ollama server.

PostgreSQL database connection status.

4.5. Robust Data Retrieval and Natural Language Processing (NLP)

User-Specific Memory Retrieval: Kaia can retrieve user preferences, facts, and interaction history from the PostgreSQL database.

Advanced Query Matching: Implemented normalize_query and match_query_category functions to process user input, allowing for more flexible and natural language queries.

Standardized Responses: Retrieval functions consistently return structured dictionaries, ensuring predictable output.

5. Development Progress Log
V1.0: Foundational System (Initial State)

Architecture: Successfully integrated Ollama and LlamaIndex to create the core RAG and command execution pipelines.

Models: Established the dual-model approach with llama2 for chat and mixtral for commands.

RAG: Implemented persistent vector indexing, persona loading, and conversational memory.

Command Execution: Built a safe and functional command execution loop.

Intelligent Routing: The main loop could differentiate between conversational queries and command requests.

V1.1: Enhanced Memory, System, and Robustness

Persistent Memory: Fully integrated PostgreSQL for storing user preferences, facts, and interaction history.

System Monitoring: Added comprehensive system status reporting via kaia_cli.py.

Improved Data Retrieval: Enhanced NLP for memory retrieval, allowing more natural phrasing.

Application Stability: Implemented robust error handling and refined logging.

V1.2: Refactoring and Security Hardening

Implemented a modular, handler-based architecture with a centralized AppState class.

Enhanced security with strict command sanitization and script allowlisting.

Added new features: script runner, video conversion, and persistent CWD tracking.

Improved reliability with robust ChromaDB initialization and comprehensive error handling.

Optimized performance with LRU caching and first-token latency tracking.

6. Future Roadmap & Planned Features
This section integrates the key suggestions from the code review into a structured and actionable plan.

Phase 1: Security and Reliability Hardening

Security Measures:

Implement a --dry-run flag for commands to allow users to preview actions.

Add authentication and collection access controls to the ChromaDB server.

Secure the KAIA_DB_PASS and add connection encryption for the database.

Implement row-level security in the PostgreSQL database.

Improved Error Handling:

Refine Kaia's ability to understand and recover from failed commands or unexpected output.

Implement command signing for critical operations.

Add timeout and health checks for Ollama model checks.

Testing:

Write a comprehensive suite of unit tests for:

Command generation and sanitization.

Database operations.

System status monitoring.

Model fallback logic.

Phase 2: Performance and Scalability

Caching and Indexing:

Implement a caching system for query results and model outputs.

Add incremental and background indexing to keep the knowledge base up-to-date without blocking the user.

Implement document versioning to track changes in the knowledge base.

Optimizations:

Implement connection pooling for the PostgreSQL database.

Batch database writes and embedding generation to improve efficiency.

Parallelize service checks to speed up startup time.

Monitoring and Observability:

Add temperature monitoring to system status reports.

Implement log rotation for the application logs.

Integrate a performance monitoring tool like Sentry for enhanced error reporting.

Phase 3: New Features and User Experience

Advanced Functionality:

Implement multi-step command sequences with an action plan generator.

Add support for multi-modal input (e.g., image understanding).

Expand tool integration to include web search, file I/O, and other capabilities.

Implement conversation history search.

User Configuration:

Move prompts to separate files for easier editing and customization.

Create a robust user configuration system to manage settings.

User Interface:

Explore a simple web UI using Gradio or Streamlit to provide a richer experience.

Add network bandwidth monitoring to system status.

7. Setup & Configuration Summary
Knowledge Base Directories:

General Knowledge: ./data

Personal Context: ./personal_context

Persona File: Kaia_Desktop_Persona.md (located in ./data)

Persistent Index Storage: ./storage (for LlamaIndex and ChromaDB)

Database: PostgreSQL (kaiadb) for structured memory.

Logging: Configured to suppress verbose library output and show critical errors.

Licensing Files: LICENSE.md (MIT License) and NOTICE.md (for third-party component licenses).
