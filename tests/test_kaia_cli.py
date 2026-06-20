import unittest
from unittest.mock import MagicMock, patch
import sys
import os
os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test_signing_secret_key_2026")
import json

# Add parent directory and core directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

from kaia_cli import KaiaCLI

class TestKaiaCLI(unittest.TestCase):
    def setUp(self):
        self.cli = KaiaCLI()

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_get_system_status_structure(self, mock_memory, mock_cpu):
        # Mock psutil returns
        mock_cpu.return_value = 10.5
        mock_mem = MagicMock()
        mock_mem.total = 16 * 1024**3
        mock_mem.available = 8 * 1024**3
        mock_mem.percent = 50.0
        mock_mem.used = 8 * 1024**3
        mock_memory.return_value = mock_mem

        status = self.cli.get_system_status()
        
        self.assertIn('timestamp', status)
        self.assertIn('cpu_info', status)
        self.assertIn('memory_info', status)
        self.assertIn('temperatures', status)
        self.assertIn('network_io', status)
        
        self.assertEqual(status['cpu_info']['usage'], 10.5)

    @patch('requests.post')
    def test_generate_command_safe(self, mock_post):
        # Mock LLM response for a safe diagnostics intent
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": '{"action": "diagnostics", "query_type": "ss", "args": ["-t", "-a"], "justification": "Check active tcp connections", "session_id": "test_session"}'
            }
        }
        mock_post.return_value = mock_response

        # Mock availability check to always return success
        with patch('utils.check_ollama_model_availability', return_value=('mistral', None)):
            command_json, error = self.cli.generate_command("show active tcp sockets")
            self.assertIsNone(error)
            parsed = json.loads(command_json)
            self.assertEqual(parsed["action"], "diagnostics")
            self.assertEqual(parsed["query_type"], "ss")

    @patch('requests.post')
    def test_generate_command_invalid_json(self, mock_post):
        # Mock LLM response returning invalid JSON
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": "Not a JSON response"
            }
        }
        mock_post.return_value = mock_response

        with patch('utils.check_ollama_model_availability', return_value=('mistral', None)):
            command, error = self.cli.generate_command("something")
            self.assertEqual(command, "")
            self.assertIn("Failed to generate structured intent", error)

if __name__ == '__main__':
    unittest.main()
