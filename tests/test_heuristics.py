import sys
import os
import logging
from unittest.mock import patch, MagicMock
import pathlib

# Set up paths relative to this script's directory
root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "core"))

import os
os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test_signing_secret_key_2026")
import config

# Mock RotatingFileHandler to prevent it from opening the file
with patch('logging.handlers.RotatingFileHandler', MagicMock()):
    from main import classify_intent_heuristically

def test_heuristics():
    test_cases = [
        ("status", "system_status"),
        ("write a story", "text_generation"),
        ("compose a poem", "text_generation"),
        ("tell me a story about robots", "text_generation"),
        ("ls -la", "command"),
    ]

    print("Running heuristic tests...")
    failed = 0
    for query, expected_action in test_cases:
        result = classify_intent_heuristically(query)
        actual_action = result['action'] if result else None
        
        if actual_action == expected_action:
            print(f"PASS: '{query}' -> {actual_action}")
        else:
            print(f"FAIL: '{query}' -> Expected {expected_action}, got {actual_action}")
            failed += 1
            
    if failed == 0:
        print("\nAll tests passed!")
        if __name__ == "__main__":
            sys.exit(0)
    else:
        print(f"\n{failed} tests failed.")
        if __name__ == "__main__":
            sys.exit(1)
        else:
            raise AssertionError(f"{failed} tests failed.")

if __name__ == "__main__":
    test_heuristics()
