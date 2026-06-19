# My Project File System Map - Kaia AI Assistant

This document provides a detailed overview of the directory structure and key files *within the `Kaia` project*, which is the primary environment for the Hardened AI Admin Agent. This structure categorizes various types of knowledge and project components for efficient access and retrieval.

The `Kaia` project resides at `/home/ekco/github/Kaia/` (also referenced as `~/github/Kaia/`).

---

## 📂 Top-Level Directories and Files

* `main.py`: The main Python script responsible for starting the conversational interface, initializing LLMs, loading personas, and managing ChromaDB indexes and database storage.
* `README.md`: The project's main README file, detailing the Hardened AI Admin Agent architecture, setup, and usage.
* `LICENSE.md`: MIT License.
* `NOTICE.md`: Third-party component licenses.
* `requirements.txt`: Python package dependencies.
* `.gitignore`: Git exclusion patterns.
* `.env`: Environment variables containing sensitive credentials (e.g. database password).

---

## 🗂️ Project Packages & Directories

### `core/`
Houses core system settings, states, CLI status monitoring, and PostgreSQL database utils.
* `__init__.py`: Package indicator.
* `config.py`: Shared project configuration, prompts, timeouts, and allowlists.
* `database_utils.py`: PostgreSQL datastore queries and operations for storing preferences, facts, and interaction history.
* `kaia_cli.py`: Retrieves system status telemetry (CPU, GPU, RAM, systemd) and generates intent payloads.
* `utils.py`: ANSI color helpers and Ollama model validation checking.

### `security/`
The hardened security subsystem, defining safety checks and execution containment.
* `__init__.py`: Package indicator.
* `cognitive_wiring.py`: Models the agent's internal emotional/affective state vectors.
* `db.py`: Security ledger database wrapper for logging incidents to `security_events.db`.
* `host_executor.py`: Safe subprocess execution layer that runs allowlisted commands, enforces write-protection boundaries, and spawns Bubblewrap sandboxes for custom scripts.
* `policy_gate.py`: Unix Domain Socket server daemon that intercepts, schema-validates, and evaluates all capability tokens and privileged intents.
* `schemas.py`: Pydantic data schemas verifying diagnostic, service control, block IP, and file writing request payloads.
* `telemetry_daemon.py`: Telemetry collector for network sockets, process listings, and systemd units.
* `telemetry_sanitizer.py`: Filters external system string fields using strict regex allowlists to prevent prompt injections.
* `threat_intel.py`: Local intelligence lookups via GeoLite2 and vulnerability correlations.

### `toolbox/`
* `video_converter.py`: Tool for converting MP4 and WebM video files to animated GIFs.

### `tests/`
The validation suite verifying all imports, command heuristics, output wrapping, and security invariants.
* `test_database_utils.py`: Database operations tests.
* `test_kaia_cli.py`: Status CLI command tests.
* `test_heuristics.py`: Fast-path classification tests.
* `verify_changes.py`: Basic imports and CLI diagnostics validator.
* `verify_security.py`: Core socket, capability token signature, fail-closed, write-protection, and sandbox masking integration tests.
* `verify_status.py`: Console formatting test.
* `verify_wrap.py`: Word-wrap formatting test.

### `scripts/`
* `activate_kaia_env.sh`: Activation shell script. Spawns necessary local services (Ollama, PostgreSQL, ChromaDB) and launches the `main.py` entrypoint.

### `data/`
Houses general knowledge documents and specialized subdirectories:
* `Kaia_Desktop_Persona.md`: Defines the core persona and operational guidelines for the Kaia AI.
* `vulkaninfo/`: Dump files containing Vulkan GPU configurations.

### `storage/`
Persists the agent's operational databases:
* `chroma_db/`: Local ChromaDB vector collection.
* `llama_index_metadata/`: Persistent indexes for quick loading.
* `security_events.db`: SQLite database for security event logs.
* `audit_ledger.json`: Append-only security audit record.
* `threat_intel/reputation.db`: SQLite database for local reputation caching.
