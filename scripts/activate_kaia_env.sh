#!/bin/bash

COLOR_GREEN="\033[92m"
COLOR_BLUE="\033[94m"
COLOR_YELLOW="\033[93m"
COLOR_RED="\033[91m"
COLOR_RESET="\033[0m"

KAIA_PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"

cd "$KAIA_PROJECT_DIR" || { echo -e "${COLOR_RED}Error: Could not navigate to project directory: $KAIA_PROJECT_DIR. Exiting.${COLOR_RESET}"; exit 1; }
mkdir -p logs

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

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo -e "${COLOR_BLUE}Checking Policy Gate Daemon status...${COLOR_RESET}"
if systemctl list-unit-files kaia-policy-gate.service &>/dev/null; then
    echo -e "${COLOR_BLUE}Systemd service kaia-policy-gate is installed.${COLOR_RESET}"
    if ! systemctl is-active --quiet kaia-policy-gate.service; then
        echo -e "${COLOR_YELLOW}Starting kaia-policy-gate service...${COLOR_RESET}"
        sudo systemctl start kaia-policy-gate.service
        sleep 2
    fi
    if systemctl is-active --quiet kaia-policy-gate.service; then
        echo -e "${COLOR_GREEN}Policy Gate Daemon systemd service is running.${COLOR_RESET}"
    else
        echo -e "${COLOR_RED}Error: Failed to start kaia-policy-gate service.${COLOR_RESET}"
        exit 1
    fi
else
    # Fallback to nohup launch
    if ! pgrep -f "security/policy_gate.py" > /dev/null; then
        echo -e "${COLOR_YELLOW}Policy Gate Daemon is not running. Starting via fallback nohup...${COLOR_RESET}"
        if [ -z "$KAIA_CAPABILITY_TOKEN_SECRET" ]; then
            echo -e "${COLOR_RED}Error: KAIA_CAPABILITY_TOKEN_SECRET environment variable is not set. Cannot start Policy Gate Daemon.${COLOR_RESET}"
            exit 1
        fi
        sudo mkdir -p /run/kaiacord 2>/dev/null || true
        sudo chown -R $USER:kaiacord /run/kaiacord 2>/dev/null || true
        sudo chmod 0770 /run/kaiacord 2>/dev/null || true
        
        nohup "$KAIA_PROJECT_DIR/.venv/bin/python" "$KAIA_PROJECT_DIR/security/policy_gate.py" > "$KAIA_PROJECT_DIR/logs/policy_gate.log" 2>&1 &
        sleep 2
        if pgrep -f "security/policy_gate.py" > /dev/null; then
            echo -e "${COLOR_GREEN}Policy Gate Daemon started successfully (fallback).${COLOR_RESET}"
        else
            echo -e "${COLOR_RED}Error: Policy Gate Daemon failed to start. Check $KAIA_PROJECT_DIR/logs/policy_gate.log for details.${COLOR_RESET}"
            exit 1
        fi
    else
        echo -e "${COLOR_GREEN}Policy Gate Daemon is already running (fallback).${COLOR_RESET}"
    fi
fi

echo -e "${COLOR_BLUE}Checking Ollama status...${COLOR_RESET}"
check_port 127.0.0.1 11434
OLLAMA_NC_STATUS=$?
if [ $OLLAMA_NC_STATUS -ne 0 ]; then
    echo -e "${COLOR_YELLOW}Ollama server not running on port 11434. Attempting to start...${COLOR_RESET}"
    nohup ollama serve > "$KAIA_PROJECT_DIR/logs/ollama.log" 2>&1 &
    OLLAMA_PID=$!
    echo -e "${COLOR_BLUE}Ollama server started with PID $OLLAMA_PID. Log: $KAIA_PROJECT_DIR/logs/ollama.log${COLOR_RESET}"
    sleep 10
    check_port 127.0.0.1 11434
    OLLAMA_NC_STATUS=$?
    if [ $OLLAMA_NC_STATUS -eq 0 ]; then
        echo -e "${COLOR_GREEN}Ollama server is now running.${COLOR_RESET}"
    else
        echo -e "${COLOR_RED}Error: Ollama server failed to start. Check $KAIA_PROJECT_DIR/logs/ollama.log for details.${COLOR_RESET}"
        exit 1
    fi
else
    echo -e "${COLOR_GREEN}Ollama server is already running.${COLOR_RESET}"
fi

echo -e "${COLOR_BLUE}All services checked. Starting Kaia CLI application...${COLOR_RESET}"

"$KAIA_PROJECT_DIR/.venv/bin/python" "$KAIA_PROJECT_DIR/main.py"

echo -e "${COLOR_BLUE}Kaia CLI application session ended.${COLOR_RESET}"
