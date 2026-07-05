import unittest
from byte_tokenizer import ByteTokenizer

class TestByteTokenizer(unittest.TestCase):
    def test_encode(self):
        sut = ByteTokenizer()
        self.assertEqual(sut.encode("hello世界😁"), [104, 101, 108, 108, 111, 228, 184, 150, 231, 149, 140, 240, 159, 152, 129])

    def test_decode(self):
        sut = ByteTokenizer()
        self.assertEqual(sut.decode([104, 101, 108, 108, 111, 228, 184, 150, 231, 149, 140, 240, 159, 152, 129]), "hello世界😁")

if __name__ == '__main__':
    unittest.main()