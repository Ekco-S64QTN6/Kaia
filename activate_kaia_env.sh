#!/bin/bash

COLOR_GREEN="\033[92m"
COLOR_BLUE="\033[94m"
COLOR_YELLOW="\033[93m"
COLOR_RED="\033[91m"
COLOR_RESET="\033[0m"

KAIA_PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd "$KAIA_PROJECT_DIR" || { echo -e "${COLOR_RED}Error: Could not navigate to project directory: $KAIA_PROJECT_DIR. Exiting.${COLOR_RESET}"; exit 1; }

source .venv/bin/activate || { echo -e "${COLOR_RED}Error: Could not activate virtual environment at $KAIA_PROJECT_DIR/.venv. Ensure it exists and is valid. Exiting.${COLOR_RESET}"; exit 1; }

echo -e "${COLOR_BLUE}Kaia's virtual environment activated and you are in $(pwd)${COLOR_RESET}"

check_port() {
    python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); result = s.connect_ex(('$1', $2)); s.close(); exit(result)"
}

echo -e "${COLOR_BLUE}Checking PostgreSQL status...${COLOR_RESET}"
if ! pg_isready -h 127.0.0.1 -p 5432 -U kaiauser > /dev/null 2>&1; then
    echo -e "${COLOR_YELLOW}PostgreSQL is not running. Attempting to start...${COLOR_RESET}"
    sudo systemctl start postgresql
    sleep 3
    if pg_isready -h 127.0.0.1 -p 5432 -U kaiauser > /dev/null 2>&1; then
        echo -e "${COLOR_GREEN}PostgreSQL started successfully.${COLOR_RESET}"
    else
        echo -e "${COLOR_YELLOW}Warning: Failed to start PostgreSQL. Kaia might not function correctly without it.${COLOR_RESET}"
    fi
else
    echo -e "${COLOR_GREEN}PostgreSQL is already running.${COLOR_RESET}"
fi

echo -e "${COLOR_BLUE}Checking ChromaDB server status...${COLOR_RESET}"
check_port 127.0.0.1 8000
CHROMA_NC_STATUS=$?
if [ $CHROMA_NC_STATUS -ne 0 ]; then
    echo -e "${COLOR_YELLOW}ChromaDB server not running on port 8000. Attempting to start...${COLOR_RESET}"
    nohup "$KAIA_PROJECT_DIR/.venv/bin/chroma" run --host 127.0.0.1 --port 8000 --path "$KAIA_PROJECT_DIR/storage/chroma_db" > "$KAIA_PROJECT_DIR/chroma.log" 2>&1 &
    CHROMA_PID=$!
    echo -e "${COLOR_BLUE}ChromaDB server started with PID $CHROMA_PID. Log: $KAIA_PROJECT_DIR/chroma.log${COLOR_RESET}"
    sleep 5
    check_port 127.0.0.1 8000
    CHROMA_NC_STATUS=$?
    if [ $CHROMA_NC_STATUS -eq 0 ]; then
        echo -e "${COLOR_GREEN}ChromaDB server is now running.${COLOR_RESET}"
    else
        echo -e "${COLOR_RED}Error: ChromaDB server failed to start. Check $KAIA_PROJECT_DIR/chroma.log for details.${COLOR_RESET}"
    fi
else
    echo -e "${COLOR_GREEN}ChromaDB server is already running.${COLOR_RESET}"
fi

echo -e "${COLOR_BLUE}Checking Ollama status...${COLOR_RESET}"
check_port 127.0.0.1 11434
OLLAMA_NC_STATUS=$?
if [ $OLLAMA_NC_STATUS -ne 0 ]; then
    echo -e "${COLOR_YELLOW}Ollama server not running on port 11434. Attempting to start...${COLOR_RESET}"
    nohup ollama serve > "$KAIA_PROJECT_DIR/ollama.log" 2>&1 &
    OLLAMA_PID=$!
    echo -e "${COLOR_BLUE}Ollama server started with PID $OLLAMA_PID. Log: $KAIA_PROJECT_DIR/ollama.log${COLOR_RESET}"
    sleep 10
    check_port 127.0.0.1 11434
    OLLAMA_NC_STATUS=$?
    if [ $OLLAMA_NC_STATUS -eq 0 ]; then
        echo -e "${COLOR_GREEN}Ollama server is now running.${COLOR_RESET}"
    else
        echo -e "${COLOR_RED}Error: Ollama server failed to start. Check $KAIA_PROJECT_DIR/ollama.log for details.${COLOR_RESET}"
        exit 1
    fi
else
    echo -e "${COLOR_GREEN}Ollama server is already running.${COLOR_RESET}"
fi

echo -e "${COLOR_BLUE}All services checked. Starting Kaia CLI application...${COLOR_RESET}"

"$KAIA_PROJECT_DIR/.venv/bin/python" llamaindex_ollama_rag.py

echo -e "${COLOR_BLUE}Kaia CLI application session ended.${COLOR_RESET}"
