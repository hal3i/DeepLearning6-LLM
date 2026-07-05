import unittest
import bpe_train as sut

class BpeTrainTest(unittest.TestCase):
    def test_count_pairs(self):
        self.assertEqual(sut.count_pairs([1, 2, 3, 1, 2]), {(1, 2): 2, (2, 3): 1, (3, 1): 1});

    def test_merge(self):
        self.assertEqual(sut.merge([1, 2, 3, 1, 2], (1, 2), 4), [4, 3, 4])

    def test_bpe_train(self):
        merge_rules = sut.train_bpe("Hello world! This is BPE raining.", vocab_size=260)
        self.assertEqual(merge_rules, {(105, 115): 256, (256, 32): 257, (105, 110): 258, (72, 101): 259})

if __name__ == '__main__':
    unittest.main()