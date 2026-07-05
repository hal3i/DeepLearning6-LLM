import unittest
from bpe_tokenizer import BPETokenizer

class TestBPETokenizer(unittest.TestCase):
    def test_encode_decode(self):
        merge_rules = {(105, 115): 256, (256, 32): 257, (105, 110): 258, (72, 101): 259}

        sut = BPETokenizer(merge_rules)

        ids = sut.encode("Hello世界😁")
        decoded = sut.decode(ids)

        self.assertEqual(ids, [259, 108, 108, 111, 228, 184, 150, 231, 149, 140, 240, 159, 152, 129])
        self.assertEqual(decoded, "Hello世界😁")

if __name__ == '__main__':
    unittest.main()