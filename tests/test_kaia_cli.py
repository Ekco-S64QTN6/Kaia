import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        # Mock LLM response for a safe command
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": "ls -la"
            }
        }
        mock_post.return_value = mock_response

        # Mock availability check to always return success
        with patch('utils.check_ollama_model_availability', return_value=('mistral', None)):
            command, error = self.cli.generate_command("list files")
            self.assertEqual(command, "ls -la")
            self.assertIsNone(error)

    @patch('requests.post')
    def test_generate_command_unsafe(self, mock_post):
        # Mock LLM response for an unsafe command
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": "rm -rf /"
            }
        }
        mock_post.return_value = mock_response

        with patch('utils.check_ollama_model_availability', return_value=('mistral', None)):
            command, error = self.cli.generate_command("delete everything")
            self.assertEqual(command, "")
            self.assertIn("Command not in allowlist", error)

if __name__ == '__main__':
    unittest.main()
