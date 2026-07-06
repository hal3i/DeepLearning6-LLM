import unittest
from char_tokenizer import CharTokenizer

class TestCharTokenizer(unittest.TestCase):
    def test_encoder(self):
        sut = CharTokenizer()
        self.assertEqual(sut.encode("hello世界😁"), [104, 101, 108, 108, 111, 19990, 30028, 128513]);

    def test_decode(self):
        sut = CharTokenizer()
        
        self.assertEqual(sut.decode([104, 101, 108, 108, 111, 19990, 30028, 128513]), "hello世界😁")

if __name__ == '__main__':
    unittest.main()