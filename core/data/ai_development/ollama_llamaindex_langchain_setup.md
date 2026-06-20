# AI Development and Local LLM Configuration

This document provides a comprehensive overview of setting up, configuring, and troubleshooting a local AI development environment, with a focus on Ollama, LlamaIndex, and LangChain.

## 1. Ollama: Installation and GPU Troubleshooting

### 1.1. Installation and Custom Model Path

To avoid filling the root partition, Ollama's models were stored in the user's home directory.

1.  **Install Ollama:**
    ```bash
    curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
    ```
2.  **Stop the Service and Create Directory:**
    ```bash
    sudo systemctl stop ollama
    mkdir -p /home/ekco/ollama/models
    ```
3.  **Configure Systemd Service:**
    Edit the service file (`sudo systemctl edit ollama.service`) to add the custom path:
    ```ini
    [Service]
    Environment="OLLAMA_MODELS=/home/ekco/ollama/models"
    ```
4.  **Fix Permissions:**
    The `ollama` user needs ownership of the models directory and traversal access to the parent directory.
    ```bash
    sudo chown -R ollama:ollama /home/ekco/ollama/models
    sudo chmod o+rx /home/ekco
    ```
5.  **Reload and Restart:**
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl start ollama
    ```

### 1.2. GPU Acceleration Troubleshooting (NVIDIA)

The primary challenge was forcing Ollama to use the NVIDIA GPU correctly and resolving subsequent permission errors on Arch Linux.

* **Challenge 1: Forcing CPU-Only Mode (Initial Workaround)**
    * **Problem:** The NVIDIA GTX 950's limited VRAM caused `cudaMalloc` errors with larger models.
    * **Solution:** Set environment variables in the systemd override file (`/etc/systemd/system/ollama.service.d/override.conf`) to hide the GPU from Ollama, forcing a fallback to the CPU.
        ```ini
        [Service]
        Environment="OLLAMA_NO_GPU=1"
        Environment="CUDA_VISIBLE_DEVICES="
        ```

* **Challenge 2: Enabling GPU Acceleration**
    * **Problem:** After installing the CUDA toolkit (`sudo pacman -S cuda`), the `ollama.service` failed to start due to permission denials and an incomplete environment.
    * **Solution Path:**
        1.  **Configure Systemd Environment:** The service file was updated to include paths to the CUDA binaries and libraries.
            ```ini
            [Service]
            Environment="PATH=/opt/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            Environment="LD_LIBRARY_PATH=/opt/cuda/lib64"
            ```
        2.  **Grant Permissions:** `ollama` user was added to `video` and `input` groups.
            ```bash
            sudo usermod -aG video,input ollama
            ```
        3.  **Set udev Rules:** Ensure proper device permissions.
            ```bash
            echo 'KERNEL=="nvidia_uvm", MODE="0666"' | sudo tee /etc/udev/rules.d/99-nvidia.rules
            sudo udevadm control --reload-rules && sudo udevadm trigger
            ```
        4.  **Install `nvidia-container-toolkit`:** Required for Docker/Podman integration with NVIDIA GPUs.
            ```bash
            yay -S nvidia-container-toolkit
            ```
        5.  **Restart Ollama:**
            ```bash
            sudo systemctl daemon-reload
            sudo systemctl restart ollama
            ```

## 3. LlamaIndex & LangChain for AI Assistant Development

### 3.1. LlamaIndex for Retrieval-Augmented Generation (RAG)

LlamaIndex is crucial for building a knowledge-aware AI assistant by providing Retrieval-Augmented Generation capabilities.

* **Key Strengths:**
    * **Data Ingestion:** Supports various data loaders (files, databases, APIs) to ingest and index diverse data.
    * **Indexing:** Creates vector stores from ingested data, enabling semantic search.
    * **Querying:** Allows natural language queries against the indexed data to retrieve relevant context.
    * **Integration:** Seamlessly integrates with Ollama (for LLMs and embeddings) and other LLM providers.

* **Implementation Approach (for Kaia):**
    1.  **Load Documents:** Use `SimpleDirectoryReader` to load Markdown files from specified knowledge base directories.
    2.  **Create/Load Index:** Build a `VectorStoreIndex` using Ollama embeddings and persist it for quick loading in subsequent sessions.
    3.  **Initialize Chat Engine:** Configure a `chat_engine` for conversational interaction, leveraging the RAG pipeline.

### 3.2. LangChain for Advanced Agentic Workflows

For building a powerful AI assistant with system interaction and persistent memory, LangChain is the recommended framework.

* **Key Strengths:**
    * **Tool Integration:** Easily wrap existing CLI functions (`run_shell_command`, `read_file`, etc.) into custom tools for the AI agent.
    * **Persistent Memory:** Offers robust memory solutions, including conversational memory and vector stores. Critically, it integrates directly with **SQL databases** for structured, long-term storage of facts, user preferences, and system reports.
    * **Flexibility:** A modular design that allows for building only the necessary components, avoiding unwanted overhead like Web3/crypto features.

* **Implementation Approach:**
    1.  Set up a LangChain project.
    2.  Define custom tools that map to desired CLI functions.
    3.  Configure a SQL database (e.g., PostgreSQL) for persistent memory.
    4.  Design and implement memory components.
    5.  Integrate your chosen LLM (e.g., via Ollama).

## 4. Future Exploration

* **Models:** Experiment with larger LLMs (`llama3`) and dedicated embedding models (`nomic-embed-text`).
* **Data:** Use LlamaIndex loaders for diverse data sources (PDFs, web pages).
* **RAG Techniques:** Explore advanced chunking, vector stores (Chroma, FAISS), and query transformations.
* **LangChain Features:** Build agents with tools, multi-step chains, and conversational memory.
* **Interface:** Create a simple web UI with Gradio or Streamlit.
