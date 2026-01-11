import sys
import os
import shutil
import time
import logging
from unittest.mock import MagicMock, patch

# Mock shutil.get_terminal_size to return a fixed width for testing
def mock_get_terminal_size(fallback=None):
    return os.terminal_size((40, 24)) # Small width to force wrapping

# Add current directory to sys.path
sys.path.append(os.getcwd())

# Mock config and other dependencies
sys.modules['config'] = MagicMock()
sys.modules['config'].COLOR_BLUE = ""
sys.modules['config'].COLOR_YELLOW = ""
sys.modules['config'].COLOR_RESET = ""

# Import the function to test, mocking logging to avoid file access
with patch('logging.handlers.RotatingFileHandler', MagicMock()):
    with patch('shutil.get_terminal_size', side_effect=mock_get_terminal_size):
        from llamaindex_ollama_rag import stream_and_print_response

class MockStream:
    def __init__(self, tokens, delay=0.0):
        self.tokens = tokens
        self.delay = delay
    
    @property
    def response_gen(self):
        for token in self.tokens:
            if self.delay > 0:
                time.sleep(self.delay)
            yield token

def test_wrap():
    print("Testing word wrap with width=40...")
    
    # Test case 1: Simple sentence that needs wrapping
    tokens = ["This ", "is ", "a ", "very ", "long ", "sentence ", "that ", "should ", "wrap ", "correctly ", "at ", "the ", "forty ", "character ", "mark ", "without ", "breaking ", "words."]
    stream = MockStream(tokens)
    
    print("\n--- Output Start ---")
    with patch('shutil.get_terminal_size', side_effect=mock_get_terminal_size):
        stream_and_print_response(stream, time.time())
    print("\n--- Output End ---")

    # Test case 2: Explicit newlines
    tokens2 = ["Line 1\n", "Line 2 is longer and should wrap if it exceeds the limit.\n", "Line 3."]
    stream2 = MockStream(tokens2)
    
    print("\n--- Output Start (Newlines) ---")
    with patch('shutil.get_terminal_size', side_effect=mock_get_terminal_size):
        stream_and_print_response(stream2, time.time())
    print("\n--- Output End ---")

    # Test case 3: Slow streaming
    tokens3 = ["This ", "should ", "appear ", "word ", "by ", "word ", "smoothly."]
    stream3 = MockStream(tokens3, delay=0.1)
    
    print("\n--- Output Start (Slow Stream) ---")
    with patch('shutil.get_terminal_size', side_effect=mock_get_terminal_size):
        stream_and_print_response(stream3, time.time())
    print("\n--- Output End ---")

if __name__ == "__main__":
    test_wrap()
