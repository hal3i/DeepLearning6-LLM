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

    def test_encode_decode_special_token(self):
        sut = BPETokenizer({(105, 115): 256, (256, 32): 257, (105, 110): 258})

        ids = sut.encode("Hello world!<|endoftext|>")
        decoded = sut.decode(ids)

        self.assertEqual(ids, [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33, 259])
        self.assertEqual(decoded, "Hello world!<|endoftext|>")

if __name__ == '__main__':
    unittest.main()