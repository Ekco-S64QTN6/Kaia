import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database_utils

class TestDatabaseUtils(unittest.TestCase):
    def test_normalize_query(self):
        self.assertEqual(database_utils.normalize_query("Hello World!"), "hello world")
        self.assertEqual(database_utils.normalize_query("  TESTing...  "), "testing")

    def test_match_query_category(self):
        self.assertEqual(database_utils.match_query_category("tell me about myself"), "about_me")
        self.assertEqual(database_utils.match_query_category("what are my preferences"), "preferences")
        self.assertEqual(database_utils.match_query_category("facts you remember"), "facts")
        self.assertEqual(database_utils.match_query_category("show our history"), "history")
        self.assertEqual(database_utils.match_query_category("random query"), "unknown")

if __name__ == '__main__':
    unittest.main()
