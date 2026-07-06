import unittest
from pretokenize import pretokenize

class TestPretokenize(unittest.TestCase):
    def test_pretokenize(self):
        self.assertEqual(pretokenize("Hello! I'm fine."), ['Hello', '!', ' I', "'m", ' fine', '.'])

if __name__ == '__main__':
    unittest.main()
