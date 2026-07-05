import unittest
import bpe_train as sut

class BpeTrainTest(unittest.TestCase):
    def test_count_pairs(self):
        self.assertEqual(sut.count_pairs([1, 2, 3, 1, 2]), {(1, 2): 2, (2, 3): 1, (3, 1): 1});

if __name__ == '__main__':
    unittest.main()