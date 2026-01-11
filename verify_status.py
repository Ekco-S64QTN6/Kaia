import sys
import os
sys.path.append('/home/ekco/github/ollama_rag_agent')
from kaia_cli import KaiaCLI

cli = KaiaCLI()
status = cli.get_system_status()
print(cli.format_system_status_output(status))
