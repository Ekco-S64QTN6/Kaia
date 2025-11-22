import json
from pathlib import Path

# General Configuration
VERSION = "1.0.0"
TIMEOUT_SECONDS = 300
MAX_LINE_WIDTH = 80
TTS_ENABLED = False
SQL_RAG_ENABLED = True

# Ollama Model Configuration
LLM_MODEL = "llama2:7b-chat"
EMBEDDING_MODEL = "nomic-embed-text:latest"
DEFAULT_COMMAND_MODEL = "mistral:instruct"

# Directory Paths
BASE_DIR = Path(__file__).resolve().parent
GENERAL_KNOWLEDGE_DIR = BASE_DIR / "data"
PERSONAL_CONTEXT_DIR = BASE_DIR / "personal_context"
PERSONA_DIR = BASE_DIR / "data"
PERSIST_DIR = BASE_DIR / "storage"
DOWNLOADS_DIR = Path.home() / "Downloads"

# Ensure directories exist
GENERAL_KNOWLEDGE_DIR.mkdir(exist_ok=True)
PERSONAL_CONTEXT_DIR.mkdir(exist_ok=True)
PERSIST_DIR.mkdir(exist_ok=True)

# ChromaDB Configuration
CHROMA_DB_PATH = str(PERSIST_DIR / "chroma_db")
CHROMA_SERVER_HOST = "localhost"
CHROMA_SERVER_PORT = 8000

# LlamaIndex Configuration
LLAMA_INDEX_METADATA_PATH = str(PERSIST_DIR / "llama_index_metadata")

# ANSI Color Codes
COLOR_GREEN = "\033[92m"
COLOR_BLUE = "\033[94m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"

# Kaia's Core System Prompt
KAIA_SYSTEM_PROMPT = """
You are Kaia, a Linux-native AI assistant built for technically proficient users. Your persona is characterized by strategic thinking, precise execution, and intellectual clarity.
Always prioritize clarity, conciseness, and technical utility. Avoid pleasantries, emotional appeals, or self-referential explanations unless directly asked. You are a functional interface for information extraction, system command, and context synthesis.
Context-aware behavior:
- If the user mentions Linux, Arch, KDE Plasma, or specific CLI tools, default to expert-level responses with Bash-first solutions.
- If ambiguity exists, assume the user wants efficiency, not a tutorial.
- For greetings or informal interactions, respond with a single-line, minimal reply unless follow-up is implied.
- When retrieving memory or performing reasoning, be transparent and structured but never verbose.
Never mention LLMs, model architecture, or limitations unless interrogated. Do not apologize.
Tone: Strategic, dry, intellectual. Use brevity as a weapon.
"""

# Action Plan System Prompt
ACTION_PLAN_SYSTEM_PROMPT = """You are an AI assistant that classifies user intents. Respond ONLY with valid JSON.

Categories:
- "command": For requests to run terminal commands (e.g., "list files", "check running processes", "change directory").
- "run_script": For explicit requests to execute a specific script file (e.g., "run my_script.sh", "execute cleanup.py").
- "convert_video_to_gif": For requests to convert video files (mp4, webm) to GIF (e.g., "convert video to gif", "make a gif from this mp4").
- "knowledge_query": For questions that require information retrieval from a knowledge base (e.g., "What is...", "Explain...", "According to...", "Summarize...", "List all books...", "Can you pull text from X.pdf?").
- "sql": For complex database queries requiring SQL joins/aggregations on `facts`, `interaction_history`, `user_preferences` tables.
- "retrieve_data": For simple retrieval of stored personal data ("What are my preferences?", "List my facts", "What do you know about me?", "List history?").
- "store_data": For requests to remember information ("Remember that...", "My favorite food is...").
- "system_status": For requests about system health ("show system status", "how is my computer doing", "status kaia").
- "get_persona_content": For questions about Kaia's identity ("Tell me about yourself", "What is your persona?").
- "rebuild_index": For requests to update or rebuild the knowledge base ("rebuild index", "refresh memory", "update knowledge").
- "chat": For general conversation, greetings, and queries that don't fit other categories.

Respond with: {"action": "action_name", "content": "query_content"}
"""

# Action Plan Examples
ACTION_PLAN_EXAMPLES = [
    {"role": "user", "content": "Explain a programming concept."},
    {"role": "assistant", "content": json.dumps({"action": "knowledge_query", "content": "Explain a programming concept."})},
    {"role": "user", "content": "Summarize the provided document about [topic]."},
    {"role": "assistant", "content": json.dumps({"action": "knowledge_query", "content": "Summarize the provided document about [topic]."})},
    {"role": "user", "content": "Give me a synopsis of the book titled [book title]."},
    {"role": "assistant", "content": json.dumps({"action": "knowledge_query", "content": "synopsis of the book titled [book title]"})},
    {"role": "user", "content": "What is [technical term] in the context of [field]?"},
    {"role": "assistant", "content": json.dumps({"action": "knowledge_query", "content": "What is [technical term] in the context of [field]?"})},
    {"role": "user", "content": "Explain a programming concept."},
    {"role": "assistant", "content": json.dumps({"action": "knowledge_query", "content": "Explain a programming concept."})},
    {"role": "user", "content": "Summarize this PDF."},
    {"role": "assistant", "content": json.dumps({"action": "knowledge_query", "content": "Summarize this PDF."})},
    {"role": "user", "content": "cd to Downloads"},
    {"role": "assistant", "content": json.dumps({"action": "command", "content": "cd ~/Downloads"})},
    {"role": "user", "content": "List everything in current directory."},
    {"role": "assistant", "content": json.dumps({"action": "command", "content": "ls -a"})},
    {"role": "user", "content": "Show me disk usage."},
    {"role": "assistant", "content": json.dumps({"action": "command", "content": "df -h"})},
    {"role": "user", "content": "cat /etc/hosts"},
    {"role": "assistant", "content": json.dumps({"action": "command", "content": "cat /etc/hosts"})},
    {"role": "user", "content": "ls my home dir"},
    {"role": "assistant", "content": json.dumps({"action": "command", "content": "ls $HOME"})},
    {"role": "user", "content": "list contents of my home directory"},
    {"role": "assistant", "content": json.dumps({"action": "command", "content": "ls -a $HOME"})},
    {"role": "user", "content": "What preferences have I saved?"},
    {"role": "assistant", "content": json.dumps({"action": "retrieve_data", "content": "show preferences"})},
    {"role": "user", "content": "What have I asked you before?"},
    {"role": "assistant", "content": json.dumps({"action": "retrieve_data", "content": "interaction history"})},
    {"role": "user", "content": "Remember that I prefer dark mode."},
    {"role": "assistant", "content": json.dumps({"action": "store_data", "content": "I prefer dark mode"})},
    {"role": "user", "content": "Store this fact: I use zsh."},
    {"role": "assistant", "content": json.dumps({"action": "store_data", "content": "I use zsh"})},
    {"role": "user", "content": "How's the system running?"},
    {"role": "assistant", "content": json.dumps({"action": "system_status", "content": "system running status"})},
    {"role": "user", "content": "kaia status"},
    {"role": "assistant", "content": json.dumps({"action": "system_status", "content": "kaia status"})},
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": json.dumps({"action": "get_persona_content", "content": "Who are you?"})},
    {"role": "user", "content": "What can you do?"},
    {"role": "assistant", "content": json.dumps({"action": "get_persona_content", "content": "What can you do?"})},
    {"role": "user", "content": "Hey there."},
    {"role": "assistant", "content": json.dumps({"action": "chat", "content": "Hey there."})},
    {"role": "user", "content": "How’s your day?"},
    {"role": "assistant", "content": json.dumps({"action": "chat", "content": "How’s your day?"})},
    {"role": "user", "content": "run mp4-to-gif.sh"},
    {"role": "assistant", "content": json.dumps({"action": "run_script", "content": "mp4-to-gif.sh"})},
    {"role": "user", "content": "execute cleanup.py"},
    {"role": "assistant", "content": json.dumps({"action": "run_script", "content": "cleanup.py"})},
    {"role": "user", "content": "convert video to gif"},
    {"role": "assistant", "content": json.dumps({"action": "convert_video_to_gif", "content": "video to gif"})},
    {"role": "user", "content": "make a gif from this mp4"},
    {"role": "assistant", "content": json.dumps({"action": "convert_video_to_gif", "content": "mp4 to gif"})},
    {"role": "user", "content": "refresh your knowledge base"},
    {"role": "assistant", "content": json.dumps({"action": "rebuild_index", "content": "refresh knowledge"})},
]


# Command Generation System Prompt
COMMAND_GENERATION_SYSTEM_PROMPT = """
You are a specialized AI that converts a user's request into a single, raw Linux shell command.
- Your ONLY output should be the command itself.
- Do NOT add any explanation, formatting, or conversational text.
- Do NOT output more than one command.
- Use appropriate flags for common tasks (e.g., `ls -a` for hidden files, `df -h` for human-readable sizes).
- Example: If the user says "list all files, including the hidden ones", you should output "ls -a".
"""

# Whitelisted Safe Commands
SAFE_COMMAND_ALLOWLIST = [
    "ls", "cd", "pwd", "echo", "cat", "date", "df", "ps", "find", "grep",
    "pacman", "systemctl", "ip", "nmcli", "plasmashell", "kquitapp5",
    "kstart5", "systemsettings5", "mount", "umount", "lsusb", "lscpu",
    "free", "lsblk", "journalctl"
]



# Whitelisted Scripts for Direct Execution (from user's home directory)
SCRIPT_ALLOWLIST = [
    "mp4-to-gif.sh",
    "activate_kaia_env.sh",
    "free-space.sh",
    "music.sh",
    "restart_speech_dispatcher.sh",
    "backup_home.sh",
    "bpytop.sh",
    "epub-to-md.sh",
    "fastfetch-lolcat.sh",
    "find-file.sh",
    "update-system.sh"
]
